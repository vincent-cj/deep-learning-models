# -*- coding: utf-8 -*-
"""
Created on 2020/10/20 下午4:25

@Project -> File: pollution-forecast-with-surrounding-cities-offline-training -> tcn.py

@Author: chenjun

@Email: cjlaso@sina.com

@Describe:
"""
import copy
import torch
import math
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size
	
	def forward(self, x):
		"""
		其实这就是一个裁剪的模块，裁剪多出来的padding, 让输出与输入的时间长度相同
		"""
		#  .contiguous() 返回一个内存连续的有相同数据的tensor，如果原tensor内存连续则返回原tensor, 进行view操作之前必须是连续的
		x = x[:, :, :-self.chomp_size].contiguous()         # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
		return x


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout = 0.2):
		"""
		卷积块
		:param n_inputs: int, 输入通道数, 每个时刻的特征数
		:param n_outputs: int, 输出通道数, 输出每个时刻的特征数
		:param kernel_size: int, 卷积核尺寸, 实际尺寸包括输入通道数, 即(kernel_size, n_inputs)
		:param stride: int, 步长，一般为1
		:param dilation: int, 膨胀系数
		:param padding: int, 填充系数
		:param dropout: float, dropout比率
		"""
		super(TemporalBlock, self).__init__()
		
		# padding左右两边对称加, 左边的padding保证输出的初始时刻有值, 右边的padding会卷积出同长度的结果, 即多出部分
		# self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
		#                                    stride = stride, padding = self.padding, dilation = dilation))
		self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
		                       stride = stride, padding = padding, dilation = dilation)
		
		# 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.net1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
	
	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
	
	def forward(self, x):
		"""
		:param x: size of (Batch, input_channel, seq_len)
		:return:
		"""
		out = self.net1(x)
		return out


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, list_channels, kernel_size = 2, dropout = 0.2):
		"""
		TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
		对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
		对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

		:param num_inputs: int， 输入通道数
		:param list_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
		:param kernel_size: int, 卷积核尺寸
		:param dropout: float, drop_out比率
		"""
		super(TemporalConvNet, self).__init__()
		
		# 卷积层的个数填满3的倍数
		res_value = len(list_channels) % 3
		if res_value > 0:
			list_channels += [list_channels[-1]] * (3 - res_value)
		
		layers = []
		num_levels = len(list_channels)
		for i in range(num_levels):
			dilation_size = 2 ** i                                       # 膨胀系数：1，2，4，8……
			in_channels = num_inputs if i == 0 else list_channels[i - 1]  # 确定每一层的输入通道数
			out_channels = list_channels[i]                               # 确定每一层的输出通道数
			
			# 通过向左填充，将卷积核最右边的单元对准当前时刻，保证当前时刻只能获取当前及之前的信息
			# padding的设置要保证输出的尺寸等于输入的尺寸，其中右边padding的部分在每一次卷积后都会去掉
			# H_out = (H_in + 2 * padding - kernel_size) / stride + 1  s.t. H_out = H_in + 1 * padding; stride=1
			# padding = kernel_size - 1
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride = 1, dilation = dilation_size,
			                         padding = (kernel_size - 1) * dilation_size, dropout = dropout)]
		
		self.blocks = []
		self._build_res_net(num_inputs, list_channels, layers)
	
	def _build_res_net(self, num_inputs, list_channels, layers):
		"""
		没3个卷积块组合成一个残差网络
		:param list_channels: 卷积的输入输出尺寸
		:param layers:
		:return:
		"""
		# list_channels的值既是输入通道数也是输出通道数
		list_channels = [num_inputs] + list_channels
		num_layers = len(list_channels)
		
		# 每三层添加一个残差层和激活层
		for i, loc in enumerate(range(0, num_layers - 1, 3)):
			net = nn.Sequential(*layers[loc: loc + 3])
			setattr(self, f'net{i}', net)
			self.blocks.append(net)
			
			if list_channels[loc] != list_channels[loc + 3]:
				downsample = nn.Conv1d(list_channels[loc], list_channels[loc + 3], 1)
			else:
				downsample = None
			relu_ = nn.ReLU()
			setattr(self, f'downsample{i}', downsample)
			setattr(self, f'relu{i}', relu_)
		
	def forward(self, x):
		"""
		输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
		这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作
		:param x: size of (Batch, input_channel, seq_len)
		:return: size of (Batch, output_channel, seq_len)
		"""
		x_2res = x.clone()
		
		for i, block in enumerate(self.blocks):
			x = block(x)
			downsample = getattr(self, f'downsample{i}')
			relu_ = getattr(self, f'relu{i}')
			if downsample is not None:
				res = downsample(x_2res)
			else:
				res = x_2res
			x = relu_(x + res)
			x_2res = x.clone()
		return x


if __name__ == '__main__':
	tcn = TemporalConvNet(14, [16, 16, 8, 8, 4])
	x = torch.randn((10, 14, 5))
	out = tcn(x)
	print(out)


