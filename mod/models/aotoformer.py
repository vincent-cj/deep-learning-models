# -*- coding: utf-8 -*-
"""
Created on 2022/1/18 下午10:33

@Project -> File: Autoformer -> aotoformer.py

@Author: chenjun

@Describe:
"""
import copy
import torch
from torch import nn
import math


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, factor=1):
        super(AutoCorrelation, self).__init__()
        self.factor = factor

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        shape=(b,h,d_k,s)
        """
        # find top k
        length = values.shape[3]
        top_k = int(self.factor * math.log(length))

        # -------- updated calculation method -------
        # shape=(b,h,d,s) -> (b,h,s)  在样本和head维度，计算的权重应该保持独立
        mean_value = torch.mean(corr, dim = 2)
        weights, index = torch.topk(mean_value, top_k, dim=-1)     # shape=(b,h,top_k)
        tmp_corr = torch.softmax(weights, dim = -1)

        # aggregation
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * tmp_corr[:, :, i].unsqueeze(dim = 2).unsqueeze(dim = 3)
        return delays_agg, tmp_corr

    def forward(self, queries, keys, values):
        # use FFT to calculate time delay correlation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)       # shape=(b,h,d_k,s)

        V, weights = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr)
        return V.permute(0, 3, 1, 2).contiguous(), weights


def clones(module, N):
    """
    Produce N identical layers.
    :param module: single layer
    :param N: nums to be copyed
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class AutoCorrelationLayer(nn.Module):
    def __init__(self, num_head, d_model, dropout=0.1, factor=1):
        """
        sublayer: compute multi head atten + resnet and layernorm
        :param num_head: num headers
        :param d_model: model normal dimension
        """
        super(AutoCorrelationLayer, self).__init__()
        # We assume d_v always equals d_k
        assert d_model % num_head == 0
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.correlation = AutoCorrelation(factor = factor)
        self.dropout = nn.Dropout(dropout)
        self.corr_weights = None
    
    def forward(self, query, key, value):
        """
        transfer to multi head and compute autocorrelation
        d_model =n_heads * d_q, len_v =len_k
        :param query: (batch_size, len_q, d_model)
        :param key: (batch_size, len_k, d_model)
        :param value: (batch_size, len_k, d_model)
        :return:
        """
        residual = query
        batch_size = query.size(0)
        
        # Do all the linear projections in batch and reshape from d_model => h, d_k
        # final shape = (b,s,h,d_k)
        query, key, value = [fc(x).view(batch_size, -1, self.num_head, self.d_k)
                             for fc, x in zip(self.linears, (query, key, value))]
        
        # Apply atten on all the projected vectors in batch. input/output shape=(b,s,h,d_k)
        corr_value, self.corr_weights = self.correlation(query, key, value)
        
        # Concat using a view and apply a final linear. shape=(b,s,d_model)
        corr_value = corr_value.view(batch_size, -1, self.num_head * self.d_k)
        corr_value = self.linears[-1](corr_value)
        
        # dropout and resnet, without layerNorm
        return self.dropout(corr_value) + residual, self.corr_weights


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        """
        sublayer: compute fc equation + resnet
        :param d_model: input dimension
        :param d_ff: output dimension
        :param dropout:
        """
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            # nn.Conv1d(in_channels = d_model, out_channels = d_ff, kernel_size = (1,), bias = True),   # 无法对非-1轴进行dropout
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
            # nn.Conv1d(in_channels = d_ff, out_channels = d_model, kernel_size = (1,), bias = True)
        )

    def forward(self, x):
        residual = x
        fc_out = self.fc(x)
        return fc_out + residual


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture, 4 blocks
    """
    def __init__(self, num_head, d_model, d_ff=None, moving_avg=25, dropout=0.1):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.correlation = AutoCorrelationLayer(num_head, d_model, dropout=dropout)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.decomp2 = SeriesDecomposition(moving_avg)

    def forward(self, x):
        """
        (b,s,d_model) -> (b,s,d_model)
        """
        x, attn = self.correlation(x, x, x)
        x, _ = self.decomp1(x)
        x = self.feed_forward(x)
        res, _ = self.decomp2(x)
        return res, attn


class WordEmbeddings(nn.Module):
    def __init__(self, max_seq_len, d_model):
        """
        word2vector embedding
        :param d_model: output dimension
        :param max_seq_len: vocabulary size or sequence length
        """
        super(WordEmbeddings, self).__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class FeatureEmbedding(nn.Module):
    def __init__(self, d_in, d_model):
        super(FeatureEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=d_in, out_channels=d_model,
                                   kernel_size=(3,), padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DecoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture, 4 blocks
    """
    def __init__(self, num_head, d_model, out_dim, d_ff=None, moving_avg=25, dropout=0.1):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_correlation = AutoCorrelationLayer(num_head, d_model, dropout=dropout)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.corss_correlation = AutoCorrelationLayer(num_head, d_model, dropout = dropout)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.out_fc = nn.Sequential(
            nn.Conv1d(in_channels = d_model, out_channels = out_dim, kernel_size = (3,), stride = (1,), padding = 1,
                      padding_mode = 'circular', bias = False),
            nn.ReLU()
        )

    def forward(self, x, cross):
        """
        (b,s,d_model) -> (b,s,d_model)
        """
        x, _ = self.self_correlation(x, x, x)
        x, trend1 = self.decomp1(x)
        x, _ = self.self_correlation(x, cross, cross)
        x, trend2 = self.decomp2(x)
        x = self.feed_forward(x)
        res, trend3 = self.decomp3(x)
        
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.out_fc(residual_trend.transpose(1, 2)).transpose(1, 2).contious()
        
        return res, residual_trend


class Encoder(nn.Module):
    def __init__(self, num_head, d_model, N, d_ff, **kwargs):
        """
        Encoder multi layer: Generic N layer decoder with masking
        :param num_head: nums of multi heads
        :param d_model: feature dimension through the whole model
        :param d_ff: project size in feedforward module
        :param dropout: probility of dropout
        :param N: nums of single encoder layer
        """
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(num_head, d_model, d_ff, **kwargs), N)
        self.norm_layer = my_Layernorm(d_model)
    
    def forward(self, x):
        """
        compute encoding result
        :param x: (batch_size, src_len, d_model)
        :return: enco_output: (batch_size, src_len, d_model)
        """
        # multi module process
        for layer in self.layers:
            x = layer(x)
        x = self.norm_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_head, d_model, out_dim, N, d_ff, **kwargs):
        """
        Encoder multi layer: Generic N layer decoder with masking
        :param num_head: nums of multi heads
        :param d_model: feature dimension through the whole model
        :param d_ff: project size in feedforward module
        :param dropout: probility of dropout
        :param N: nums of single encoder layer
        """
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(num_head, d_model, out_dim, d_ff, **kwargs), N)
        self.norm_layer = my_Layernorm(d_model)
        self.final_fc = nn.Linear(d_model, out_dim, bias=True)
    
    def forward(self, x, cross, trend=None):
        """
        compute encoding result
        :param x: (batch_size, tgt_len, d_model)
        :param cross: (batch_size, src_len, d_model)
        :param trend:
        :return: enco_output: (batch_size, src_len, d_model)
        """
        
        # multi module process
        for layer in self.layers:
            x, residual_trend = layer(x, cross)
            trend = trend + residual_trend if trend is not None else trend
        x = self.norm_layer(x)
        x = self.final_fc(x)
        
        return x, trend


class Autoformer(nn.Module):
    def __init__(self, num_head, d_model, d_ff, N, out_dim, enco_seq_len = None, deco_seq_len = None, enc_input_dim=None,
                 dec_input_dim = None, moving_avg = 25, dropout = 0.1, is_classification = True, *args, **kwargs):
        """
        Transformer model
        :param enco_seq_len: max sequence length or vocabulary length of endocer input
        :param deco_seq_len: max sequence length or vocabulary length of dedocer input
        :param out_dim: output or target feature dimension
        :param num_head: num of multi heads in attention
        :param d_model: normal feature dimension if most cases of Transformer
        :param d_ff: feature dimension in FeedForward Layer
        :param N: num of EncoderLayers or DecoderLayers
        :param dropout: dropout ration
        :param is_classification: classification if True else regression
        """
        super(Autoformer, self).__init__()
        
        if not ((enco_seq_len & deco_seq_len) | (enc_input_dim & dec_input_dim)):
            raise ValueError('either word embedding or feature projection is needed.')
        
        self.model_conf = {
            'enco_seq_len': enco_seq_len,
            'deco_seq_len': deco_seq_len,
            'enc_input_dim': enc_input_dim,
            'dec_input_dim': dec_input_dim,
            'out_dim': out_dim,
            'num_head': num_head,
            'd_model': d_model,
            'd_ff': d_ff,
            'N': N,
            'dropout': dropout,
            'classification': is_classification,
            **kwargs
        }

        self.decomp = SeriesDecomposition(moving_avg)
        
        if enc_input_dim & dec_input_dim:
            self.enc_emb = FeatureEmbedding(enc_input_dim, d_model)
            self.dec_emb = FeatureEmbedding(dec_input_dim, d_model)
        else:
            self.enc_emb = WordEmbeddings(enco_seq_len, d_model)
            self.dec_emb = WordEmbeddings(deco_seq_len, d_model)
        
        self.encoder = Encoder(num_head, d_model, N, d_ff, **kwargs)
        self.decoder = Decoder(num_head, d_model, out_dim, N, d_ff, **kwargs)
        self.final_fc = nn.Linear(d_model, out_dim)
        self.classification = is_classification
    
    def forward(self, enco_inputs, y_prevs):
        """
        
        :param enco_inputs: [batch_size, src_len] or [batch_size, src_len, feature_len]
        :param y_prevs: [batch_size, tgt_len] or [batch_size, tgt_len, feature_len]
        :return:
        """
        # -------- decoder input intialize -------
        enc_mean = torch.mean(enco_inputs, dim = 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(*y_prevs.shape).to(enco_inputs.device)
        seasonal_init, trend_init = self.decomp(enco_inputs)
        
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], enc_mean], dim = 1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim = 1)
        
        # enc_outputs: (batch_size, src_len, d_model)
        enco_inputs = self.enc_emb(enco_inputs)
        enc_outputs = self.encoder(enco_inputs)

        # decoding
        dec_inputs = self.dec_emb(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_inputs, enc_outputs, trend = trend_init)
        
        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


