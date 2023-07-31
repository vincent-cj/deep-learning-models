# -*- coding: utf-8 -*-
"""
Created on 2023/7/24 下午9:11

@Project -> File: deep-learning-models -> swin_transformer.py

@Author: vincent-cj

@Describe:

(B,C,H,W) -> (B, C, H/Mh, Mh, W/Mw, Mw)
在VIT中，(B, C, H/Mh, Mh, W/Mw, Mw) -> (B, Nw, C*Mh*Mw)，即将每个窗口的元素当作特征与通道合并，在窗口之间进行atten操作
在swin-transformer中，(B, C, H/Mh, Mh, W/Mw, Mw) -> (B*Nw, Mh*Mw, C)，即在窗口内部进行atten操作
"""
import torch
from torch import nn
import torch.nn.functional as F


class DropPath(nn.Module):
	def __init__(self, drop_prob):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob
	
	def forward(self, x):
		if self.drop_prob == 0. or not self.training:
			return x
		shape = (x.shape[0],) + (x.ndim - 1) * (1,)
		keep_prob = 1 - self.drop_prob
		rand_state = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
		rand_state.floor_()
		x = x.div(keep_prob) * rand_state  # 部分样本置为0，除以操作是为了平衡大小
		return x


class PatchEmbedding(nn.Module):
	"""
	通过4*4的窗口将特征图尺寸进行4倍下采样，并增加通道数为embed_dim
	"""
	
	def __init__(self, in_dim = 3, patch_size = 4, embed_dim = 96, norm_layer = None):
		super(PatchEmbedding, self).__init__()
		patch_size = (patch_size, patch_size)
		self.patch_size = patch_size
		self.in_dim = in_dim
		self.embed_dim = embed_dim
		self.conv = nn.Conv2d(in_dim, embed_dim, patch_size, patch_size)
		self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity
	
	def forward(self, x: torch.Tensor):
		"""除了本层，其他层的输入都是三维数组"""
		_, _, H, W = x.shape
		
		# 为适配不同尺寸的图像输入，需要判断并进行padding操作
		if (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0):
			# 从最后一个维度指定双向padding的吃吃
			x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
			              0, self.patch_size[0] - H % self.patch_size[0]))
		
		x = self.conv(x)
		# 因为后续使用时都是三维数组，为了记录原始特征图的吃吃，需要返回H,W
		_, _, H, W = x.shape
		
		# 因为后续的MSA操作是在切片后的特征图上进行的，因此需要将特征图展平并放在dim==1的位置，作为token使用
		# (b,c,h,w) -> (b,c,h*w) -> (b,h*w,c)
		x = x.flatten(2).transpose(1, 2)
		return x, H, W


class PatchMerging(nn.Module):
	"""
	通过将特征图上，横向和纵向隔开位置的像素点组合在一起的方式，形成四张特征图
	对其进行layernorm操作和线性操作，实现通道数翻倍、特征图尺寸减半的
	每个layer中，MSA操作不会改变通道数和特征图尺寸，layer后跟着的patchmerging会进行此操作
	"""
	
	def __init__(self, dim, norm = nn.LayerNorm):
		super(PatchMerging, self).__init__()
		self.liner = nn.Linear(4 * dim, 2 * dim, bias = False)
		self.norm = norm(4 * dim)
	
	def forward(self, x, H, W):
		B, L, C = x.shape
		assert L == H * W
		
		x = x.reshape(B, H, W, C)
		if (H % 2) or (W % 2):
			x = torch.nn.functional.pad(x, (0, 0, 0, W % 2, 0, H % 2))
		
		x0 = x[:, ::2, ::2, :]  # [B, H/2, W/2, C] 横纵都减半的特征图
		x1 = x[:, ::2, 1::2, :]
		x2 = x[:, 1::2, ::2, :]
		x3 = x[:, 1::2, 1::2, :]
		
		x = torch.cat((x0, x1, x2, x3), dim = -1)  # [B, H/2, W/2, 4C]
		x = self.norm(x)  # 对每个像素点，在所有通道上归一化，实际上是原图中相邻4个点向量一起进行归一化
		x = self.liner(x)  # [B, H/2, W/2, 2C]
		_, H, W, _ = x.shape
		x = x.reshape(B, -1, self.liner.out_features)
		return x, H, W


class WindowAttention(nn.Module):
	"""需要加上相对位置偏量"""
	
	def __init__(self, num_dim, num_heads, window_size, qkv_bias = False, atten_drop = 0., proj_drop = 0.):
		super(WindowAttention, self).__init__()
		assert num_dim % num_heads == 0, 'num_heads does not match num_dim'
		self.num_heads = num_heads
		self.qkv = nn.Linear(num_dim, 3 * num_dim, bias = qkv_bias)
		self.scale = num_dim ** 0.5
		self.attn_drop = nn.Dropout(atten_drop)
		self.proj = nn.Linear(num_dim, num_dim)
		self.proj_drop = nn.Dropout(proj_drop)
		
		# 初始化相对位置信息
		if isinstance(window_size, int):
			window_size = [window_size, window_size]
		assert len(window_size) == 2
		
		# -------- 创建相对位置，针对每个窗口 -------
		coord_h = torch.arange(window_size[0])
		coord_w = torch.arange(window_size[1])
		coords = torch.stack(torch.meshgrid([coord_h, coord_w])).flatten(1)  # (2, Ws*Ws)
		
		# (2, Ws*Ws, Ws*Ws), 2分别代表h方向和w方向的坐标
		relative_coords = coords[:, :, None] - coords[:, None, :]  # 每个位置的坐标减去其他位置的坐标
		relative_coords[0, :, :] += window_size[0] - 1  # 将所有负数变为非负数
		relative_coords[1, :, :] += window_size[1] - 1
		
		# 为了区分横纵坐标，将横坐标加上“列长度”
		# 因为相对位置最小为-M + 1, 最大为M-1，加上0，总共 2M-1个元素
		relative_coords[0, :, :] *= 2 * window_size[1] - 1
		relative_coords = relative_coords.sum(0)  # (Ws*Ws, Ws*Ws)
		self.register_buffer('relative_coords', relative_coords)  # 固定值，不参与训练
		
		# 横纵坐标最大取值各有2m-1个元素，为了增加head间的多样性，每个head不共享
		self.relativee_pos_bias = nn.Parameter(torch.zeros(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)))
		nn.init.trunc_normal_(self.relativee_pos_bias, std = 0.2)
	
	def forward(self, x, mask = None):
		"""
		x.shape = (B*Nw, Ws*Ws, C)
		mask.shape = (Nw, Ws*Ws, Ws*Ws), 布尔类型
		"""
		#
		B_, L, C = x.shape
		# (3, B*Nw, num_heads, Ws*Ws, head_dim)
		qkv = self.qkv(x).reshape(B_, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)
		
		# (B*Nw, num_heads, Ws*Ws, head_dim) @ (B*Nw, num_heads, head_dim, Ws*Ws)
		# -> (B*Nw, num_heads, Ws*Ws, Ws*Ws)
		attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
		
		# 为所有head的窗口加上位置信息, (Ws*Ws, Ws*Ws, num_heads) -> (num_heads,Ws*Ws, Ws*Ws)
		relative_pos_bis = self.relativee_pos_bias[self.relative_coords].permute(2, 0, 1).contiguous()
		attn += relative_pos_bis[None]  # (B*Nw, num_heads, Ws*Ws, Ws*Ws)
		
		# 在shifted-window MSA中为整个特征图增加mask
		if mask is not None:
			Nw = mask.shape[0]
			attn = attn.reshape(-1, Nw, self.num_heads, L, L)
			attn = attn.masked_fill_(mask.unsqueeze(0).unsqueeze(2), -100)
			attn = attn.reshape(B_, self.num_heads, L, L)
		attn = F.softmax(attn, dim = -1)
		attn = self.attn_drop(attn)
		
		# (B*Nw, num_heads, Ws*Ws, head_dim) -> (B*Nw, Ws*Ws, num_heads, head_dim)
		# -> (B*Nw, Ws*Ws, num_dim)
		x = (attn @ v).transpose(1, 2).flatten(2)
		x = self.proj_drop(self.proj(x))
		return x


def create_atten_mask(x, H, W, window_size, shift_size):
	"""在特征图尺寸和窗口尺寸相同的层中，mask均可共享"""
	Hp = (window_size - H % window_size) % window_size + H
	Wp = (window_size - W % window_size) % window_size + W
	
	# 原始特征图平移后切分为小窗口，每个小窗口的mask不一样
	att_mask = torch.zeros(1, Hp, Wp, 1, device = x.device)
	
	# 将特征图进行平移，此时可分为几个小块，每个小块都是连续的
	h_slices = [slice(0, -window_size),
	            slice(-window_size, -shift_size),
	            slice(-shift_size, None)]
	w_slices = [slice(0, -window_size),
	            slice(-window_size, -shift_size),
	            slice(-shift_size, None)]
	
	# 将每个小块表上不同的编号，表示连续的区域
	flag = 0
	for h in h_slices:
		for w in w_slices:
			att_mask[:, h, w, :] = flag
			flag += 1
	
	# 将mask划分为相等的窗口
	att_mask = att_mask.reshape(1, Hp // window_size, window_size,
	                            Wp // window_size, window_size, 1)
	att_mask = att_mask.reshape(-1, window_size * window_size)
	
	# 此时已将窗口内的像素点平铺，统计每个像素点与所有其他像素点是否在同一个连续区域
	# (Nw, Ws*Ws, 1)-(Nw, 1, Ws*Ws) -> (Nw, Ws*Ws, Ws*Ws)
	att_mask = att_mask.unsqueeze(2) - att_mask.unsqueeze(1)
	att_mask = att_mask.bool()  # 相等的位置相减为0，在同一个区域，置为False
	return att_mask


class MLP(nn.Module):
	def __init__(self, in_dim, hidden_dim = None, out_dim = None, act_layer = nn.GELU, drop_ratio = 0.):
		super(MLP, self).__init__()
		hidden_dim = hidden_dim or in_dim
		out_dim = out_dim or in_dim
		self.fc1 = nn.Linear(in_dim, hidden_dim)
		self.act = act_layer()
		self.drop1 = nn.Dropout(drop_ratio)
		self.fc2 = nn.Linear(hidden_dim, out_dim)
		self.drop2 = act_layer()
	
	def forward(self, x):
		"""x.shape=B, L, C"""
		x = self.drop1(self.act(self.fc1(x)))
		x = self.drop2(self.fc2(x))
		return x


class SwinTransformerBlock(nn.Module):
	"""
	一个transfomer层，在整个block中，输入输出都是三维张量，特征图尺寸和通道数都不变
	"""
	
	def __init__(self, in_dim, num_heads, window_size = 7, shift_size = 0, qkv_bias = False,
	             proj_drop = 0., atten_drop = 0., drop_path = 0., mlp_ratio: int = 4,
	             norm_layer = nn.LayerNorm):
		super(SwinTransformerBlock, self).__init__()
		assert shift_size < window_size, 'shift size should be lg windo size.'
		self.norm1 = norm_layer(in_dim)
		self.norm2 = norm_layer(in_dim)
		self.window_size = window_size
		self.shift_size = shift_size
		self.atten = WindowAttention(in_dim, num_heads, window_size, qkv_bias = qkv_bias,
		                             atten_drop = atten_drop, proj_drop = proj_drop)
		
		self.drop_path = DropPath(drop_path)
		self.mlp = MLP(in_dim, in_dim * mlp_ratio)
	
	def forward(self, x, H, W, attn_mask):
		"""
		attn_mask.shape=(Nw, Ws*Ws, Ws*Ws), 布尔类型
		"""
		B, L, C = x.shape
		assert L == H * W, 'feature size error.'
		shortcut = x
		x = self.norm1(x)
		
		# W-MSA之前要进行窗口划分，检查并填充至窗口的整倍数
		x = x.reshape(B, H, W, C)
		pad_w = (self.window_size - W % self.window_size) % self.window_size
		pad_h = (self.window_size - H % self.window_size) % self.window_size
		if pad_h or pad_w:
			x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
		_, Hp, Wp, _ = x.shape
		
		# 在WS-MSA中，需要移动特征图
		if self.shift_size > 0:
			
			# 在SW-MSA中，将特征图上面和左面的一部分，分别移动到下面和右面
			x = torch.roll(x, shifts = (-self.shift_size, -self.shift_size), dims = (1, 2))
		else:
			attn_mask = None
		
		# 将特征图切分为小窗口
		x = x.reshape(B, Hp // self.window_size, self.window_size,
		              Wp // self.window_size, self.window_size, C)
		# (B*Nw, Ws*Ws, C)
		x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size * self.window_size, C)
		
		# 将小窗口的像素点平铺，以之为token，进行MAS操作
		x = self.atten(x, attn_mask)
		
		# 将小窗口恢复为大的特征图
		x = x.reshape(B, Hp // self.window_size, Wp // self.window_size,
		              self.window_size, self.window_size, C)
		x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
		
		# 将特征图移动回来
		if self.shift_size > 0:
			x = torch.roll(x, shifts = (self.shift_size, self.shift_size), dims = (1, 2))
		
		x = x[:, :H, :W, :C].reshape(B, -1, C)
		
		# 残差思想，同时加入drop path，让部分样本在MSA分支失效
		x = shortcut + self.drop_path(x)
		x = x + self.drop_path(self.mlp(self.norm2(x)))
		return x


class BasicStage(nn.Module):
	"""Swin Transformor分为好几个stage，每个stage由若干层block和一个patch merging组成（最后一个stage除外）"""
	def __init__(self, in_dim, depth, num_head, window_size, mlp_ratio=4, qkv_bias=True, proj_drop=0.,
	             atten_drop=0, drop_path=0., norm_layer=nn.LayerNorm, downsample: PatchMerging = None):
		super(BasicStage, self).__init__()
		shift_size = window_size // 2
		self.window_size = window_size
		self.shift_size = shift_size
		
		self.blocks = nn.ModuleList([
			SwinTransformerBlock(in_dim, num_head, window_size,
			                     shift_size = 0 if (i % 2 == 0) else shift_size,  # 偶数层使用shifted window
			                     qkv_bias = qkv_bias,
			                     proj_drop = proj_drop,
			                     atten_drop = atten_drop,
			                     drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
			                     mlp_ratio = mlp_ratio,
			                     norm_layer = norm_layer)
			for i in range(depth)
		])
		
		if downsample is None:
			self.downsample = None
		else:
			self.downsample = downsample(in_dim, norm_layer)    # patch merging
	
	def forward(self, x, H, W):
		# (Nw, Ws*Ws, Ws*Ws)
		atten_mask = create_atten_mask(x, H, W, self.window_size, self.shift_size)
		
		for blk in self.blocks:
			x = blk(x, H, W, atten_mask)
		
		if self.downsample is not None:
			x, H, W = self.downsample(x, H, W)
		
		return x, H, W
		

class SwinTransformer(nn.Module):
	def __init__(self, patch_size=4, in_dim=3, num_class=6, embed_dim=96,
	             depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=7,
	             mlp_ratio=4, qkv_bias=True, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.1,
	             norm_layer=nn.LayerNorm, patch_norm=True):
		super(SwinTransformer, self).__init__()
		
		num_stages = len(depths)
		self.out_dim = int(embed_dim * 2 ** (num_stages - 1))
		self.patch_embed = PatchEmbedding(in_dim, patch_size, embed_dim,
		                                  norm_layer if patch_norm else None)
		self.pos_drop = nn.Dropout(drop_ratio)
		drop_path_ratios = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]
		drop_path_ratios = torch.split(torch.tensor(drop_path_ratios), depths)
		
		self.stages = nn.ModuleList()
		for i in range(num_stages):
			stage = BasicStage(in_dim = embed_dim * 2 ** i, depth = depths[i],
			                    num_head = num_heads[i],
			                    window_size = window_size,
			                    mlp_ratio = mlp_ratio,
			                    qkv_bias = qkv_bias,
			                    proj_drop = drop_ratio,
			                    drop_path = drop_path_ratios[i].tolist(),
			                    norm_layer = norm_layer,
			                    downsample = PatchMerging if i != num_stages - 1 else None  # 最后一个stage不进行下采样
			                    )
			self.stages.append(stage)
		self.norm = norm_layer(self.out_dim)
		self.avgpool = nn.AdaptiveAvgPool1d(1)
		self.head = nn.Linear(self.out_dim, num_class) if num_class > 0 else nn.Identity()
		self.apply(self._init_weights)
	
	@staticmethod
	def _init_weights(m):
		if isinstance(m, nn.Linear):
			nn.init.trunc_normal_(m.weight, std = 0.2)
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, nn.LayerNorm):
			nn.init.zeros_(m.bias)
			nn.init.ones_(m.weight)
		elif isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight)
			if m.bias is not None:
				nn.init.zeros_(m.bias)
				
	def forward(self, x):
		# x.shape = b, c, h, w
		x, H, W = self.patch_embed(x)
		x = self.pos_drop(x)
		
		for stage in self.stages:
			x, H, W = stage(x, H, W)
		
		x = self.norm(x)    # [B,L,C]
		x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
		x = torch.flatten(x, 1)
		x = self.head(x)
		return x


if __name__ == '__main__':
	model = SwinTransformer()
	x = torch.rand((4, 3, 221, 224))    # 理论上可接受任何尺寸
	out = model(x)
	print(out.shape)

