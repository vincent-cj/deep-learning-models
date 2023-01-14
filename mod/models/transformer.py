# -*- coding: utf-8 -*-
"""
Created on 2021/9/11 下午4:32

@Project -> File: learning_models -> transformer.py

@Author: chenjun

@Describe:
"""
import numpy as np
import math
import torch
import copy
import random
from torch.utils import data as Data
from torch import nn
from torch import optim


# Transformer Parameters
d_model = 512       # Embedding Size
d_ff = 2048         # FeedForward dimension
d_k = d_v = 64      # dimension of K(=Q), V
n_layers = 6        # number of Encoder of Decoder Layer
n_heads = 8         # number of heads in Multi-Head Attent


def attention(query, key, value, atten_mask=None):
    """
    Compute Scaled Dot Product Atten
    should: d_q == d_k, len_v == len_q, atten_mask.shape equal to scores or can be broadcast
    :param query: (batch_size, n_heads, len_q, d_q)
    :param key: (batch_size, n_heads, len_k, d_q)
    :param value: (batch_size, n_heads, len_k, d_v)
    :param atten_mask: equal or can be broadcast to (batch_size, n_heads, len_q, len_k)
    :return: atten_value: (..., len_q, d_v)
    """
    d_k = query.size(-1)
    # (..., len_q, d_q) * (..., d_q, len_q) -> (..., len_q, len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # fill the scores as -inf where atten_mask is True, so as to ignore these values when compute atten_prob
    # there are two atten_mask，pad_mask and pos_mask, in transformer.
    # 1st is to shield the padding position, which is used to fill all the samples to the same length.
    # 2nd is to shield the future position, so as to avoid a certain step seeing the future information.
    if atten_mask is not None:
        scores = scores.masked_fill(atten_mask, -1e9)
    atten_prob = torch.softmax(scores, dim = -1)
    
    # (..., len_q, len_k) * (..., len_k, d_v) -> (..., len_q, d_v)
    atten_value = torch.matmul(atten_prob, value)
    
    return atten_value, atten_prob


def clones(module, N):
    """
    Produce N identical layers.
    :param module: single layer
    :param N: nums to be copyed
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_model):
        """
        sublayer: compute multi head atten + resnet and layernorm
        :param num_head: num headers
        :param d_model: model normal dimension
        """
        super(MultiHeadAttention, self).__init__()
        # We assume d_v always equals d_k
        assert d_model % num_head == 0
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.corr_weights = None
        self.layerNorm = nn.LayerNorm(d_model)
        self.correlation = attention
    
    def forward(self, query, key, value, atten_mask = None):
        """
        transfer to multi head and compute atten
        d_model =n_heads * d_q, len_v =len_k
        :param query: (batch_size, len_q, d_model)
        :param key: (batch_size, len_k, d_model)
        :param value: (batch_size, len_k, d_model)
        :param atten_mask: (batch_size, len_q, len_k)
        :return:
        """
        residual = query
        
        # Same mask applied to all h heads, (batch_size, len_q, len_k) -> (batch_size, 1, len_q, len_k)
        if atten_mask is not None:
            atten_mask = atten_mask.unsqueeze(1)
        batch_size = query.size(0)
        
        # Do all the linear projections in batch and reshape from d_model => h, d_k
        query, key, value = [fc(x).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
                             for fc, x in zip(self.linears, (query, key, value))]
        
        # Apply atten on all the projected vectors in batch.
        corr_value, self.corr_weights = self.correlation(query, key, value, atten_mask = atten_mask)
        
        # Concat using a view and apply a final linear.
        corr_value = corr_value.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        corr_value = self.linears[-1](corr_value)
        
        # resnet and layernorm
        return self.layerNorm(corr_value + residual), self.corr_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.):
        """
        sublayer: compute fc equation + resnet and layernorm
        :param d_model: input dimension
        :param d_ff: output dimension
        :param dropout:
        """
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        fc_out = self.fc(x)
        return self.layerNorm(fc_out + residual)


class EncoderLayer(nn.Module):
    def __init__(self, num_head, d_model, d_ff, dropout=0.):
        """
        single layer: Encoderlayer is made up of self-attn and feed forward (defined below)
        :param num_head:
        :param d_model:
        :param dropout:
        :param dropout:
        """
        super(EncoderLayer, self).__init__()
        self.enco_self_attn = MultiHeadAttention(num_head, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, enco_inputs, enco_self_attn_mask):
        """
        
        :param enco_inputs: (batch_size, src_len, d_model)
        :param enco_self_attn_mask: (batch_size, src_len, src_len) or (batch_size, 1, src_len)
        :return:
        """
        # enco_inputs: (batch_size, src_len, d_model), attn: (batch_size, n_heads, src_len, src_len)
        atten_value, attn = self.enco_self_attn(enco_inputs, enco_inputs, enco_inputs, enco_self_attn_mask)
        enc_outputs = self.feed_forward(atten_value)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class Embeddings(nn.Module):
    def __init__(self, max_seq_len, d_model):
        """
        word2vector embedding
        :param d_model: output dimension
        :param max_seq_len: vocabulary size or sequence length
        """
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 500):
        """
        get positional encoding to indicates the relative or absolute position
        :param d_model: feature dimension size
        :param max_len: maximum sequence length
        :return pe: (1, max_len, d_model)
        """
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        dimension = torch.arange(0, d_model)
        
        # pe: (max_len, d_model)
        pe = position / np.power(10000, 2 * (dimension // 2) / d_model)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def get_attn_pad_mask(key):
    """
    # pad mask is to shield the padding position, which is used to fill all the samples to the same length.
    :param key: [batch_size, len_k], information before embedding
    :return: pad_mask: (batch_size, 1, len_k) : means all the queries positions are the same, while every sample is different
    """
    # position equal to 0 is padding token, of which True is to be masked
    pad_mask = key.detach().eq(0).unsqueeze(1)
    return pad_mask


def get_attn_pos_mask(seq_len):
    """
    # pos mask is to shield the future position, so as to avoid a certain step seeing the future information.
    :param seq_len: sequence length
    :return: pos_mask: (1, seq_len, seq_len) : means all samples are the same, while every query position is different
    """
    pos_mask = torch.triu(torch.ones(seq_len, seq_len)).t() == 0
    pos_mask = pos_mask.unsqueeze(0)
    return pos_mask


class DecoderLayer(nn.Module):
    def __init__(self, num_head, d_model, d_ff, dropout=0.):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(num_head, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dec_enc_attn = MultiHeadAttention(num_head, d_model)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        compute decoding result
        :param dec_inputs: (batch_size, tgt_len, d_model)
        :param enc_outputs: (batch_size, src_len, d_model)
        :param dec_self_attn_mask: (batch_size, tgt_len, tgt_len)
        :param dec_enc_attn_mask: (batch_size, tgt_len, src_len) or (batch_size, 1, src_len)
        :return:
        """
        # dec_outputs: (batch_size, tgt_len, d_model), dec_self_attn: (batch_size, n_heads, tgt_len, tgt_len)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        
        # dec_outputs: (batch_size, tgt_len, d_model), dec_enc_attn: (batch_size, h_heads, tgt_len, src_len)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        
        dec_outputs = self.feed_forward(dec_outputs)    # (batch_size, tgt_len, d_model)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, num_head, d_model, d_ff, N, dropout = 0.1, max_seq_len = None, add_pe = False, *args, **kwargs):
        """
        Encoder multi layer: Generic N layer decoder with masking
        :param num_head: nums of multi heads
        :param d_model: feature dimension through the whole model
        :param d_ff: project size in feedforward module
        :param dropout: probility of dropout
        :param N: nums of single encoder layer
        """
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(num_head, d_model, d_ff, dropout), N)
        self.get_attn_pad_mask = get_attn_pad_mask
        self.src_emb = Embeddings(max_seq_len, d_model) if max_seq_len is not None else None
        self.add_pe = PositionalEncoding(d_model) if add_pe else None
    
    def forward(self, x):
        """
        compute encoding result
        :param x: (batch_size, src_len) or (batch_size, src_len, d_model)
        :return: enco_output: (batch_size, src_len, d_model)
        """
        # preprocessing: embedding and add positional embedding
        if self.src_emb is not None:
            if len(x.shape) == 2:
                x = self.src_emb(x)
            else:
                raise Exception('source embedding should not exist when input.shape is not 2')
        
        if self.add_pe is not None:
            x = self.add_pe(x)
        
        # get pad mask based on origin vector
        # (batch_size, 1, src_len)
        if self.get_attn_pad_mask is not None:
            pad_mask = self.get_attn_pad_mask(x).to(x.device)
        else:
            pad_mask = None
        
        # multi module process
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x, pad_mask


class Decoder(nn.Module):
    def __init__(self, h, d_model, d_ff, N, dropout=0.1, max_seq_len=None, add_pe=False):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(h, d_model, d_ff, dropout), N)
        self.tgt_emb = Embeddings(max_seq_len, d_model) if max_seq_len is not None else None
        self.add_pe = PositionalEncoding(d_model) if add_pe else None
    
    def forward(self, deco_input, src_pad_mask, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        :param deco_input: (batch_size, tgt_len) or (batch_size, tgt_len, d_model)
        :param src_pad_mask: (batch_size, tgt_len, src_len) or (batch_size, 1, src_len)
        :param enc_outputs: (batsh_size, src_len, d_model)
        :return:
        """

        # preprocessing: embedding and add positional embedding
        pos_mask = get_attn_pos_mask(deco_input.size(1)).to(deco_input.device)
        
        if self.tgt_emb is not None:
            deco_input = self.tgt_emb(deco_input)
        if self.add_pe is not None:
            deco_input = self.add_pe(deco_input)

        # multi module process
        deco_output = deco_input
        for layer in self.layers:
            deco_output, _, _ = layer(deco_output, enc_outputs, pos_mask, src_pad_mask)
        return deco_output


class Transformer(nn.Module):
    def __init__(self, out_dim, num_head, d_model, d_ff, N, enco_seq_len=None, deco_seq_len=None,
                 dropout=0., prob=0., is_classification=True, *args, **kwargs):
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
        :param prob: the probability for step i+1 import ground truth of step i+1
        :param is_classification: classification if True else regression
        """
        super(Transformer, self).__init__()
        self.model_conf = {
            'enco_seq_len': enco_seq_len,
            'deco_seq_len': deco_seq_len,
            'out_dim': out_dim,
            'num_head': num_head,
            'd_model': d_model,
            'd_ff': d_ff,
            'N': N,
            'dropout': dropout,
            'prob': prob,
            'classification': is_classification
        }
        
        self.encoder = Encoder(num_head, d_model, d_ff, N, dropout=dropout, max_seq_len = enco_seq_len)
        self.decoder = Decoder(num_head, d_model, d_ff, N, dropout=dropout, max_seq_len = deco_seq_len)
        self.final_fc = nn.Linear(d_model, out_dim)
        self.prob = prob
        self.classification = is_classification

    def forward(self, enco_inputs, y_prevs, parallel: bool, is_train: bool):
        """
        in case of inference, deco_inputs can be [batch_size, 1], due to the future target is unvailable.
        :param enco_inputs: [batch_size, src_len]
        :param y_prevs: [batch_size, tgt_len] or [batch_size, 1]
        :param parallel: if True, then decoder goes parallel bettween different time steps, else one by one.
        :param is_train: train mode if is_train is True or test mode
        :return:
        """
        # enc_outputs: (batch_size, src_len, d_model), src_pad_mask: (batch_size, src_len, src_len)
        enc_outputs, src_pad_mask = self.encoder(enco_inputs)

        # greedy decoding, teacher forcing
        if parallel:
            dec_outputs = self.decoder(y_prevs, src_pad_mask, enc_outputs)
            y_values = self.final_fc(dec_outputs)            # value
            y_probs = torch.softmax(y_values, dim = -1)       # probability
            y_clss = torch.argmax(y_probs, dim = -1)          # classes

            if self.classification is True:
                return y_probs, y_clss
            else:
                return y_values
            
        # step i+1 import decoding result of step i
        else:
            out = torch.tensor([]).to(y_prevs.device)
            ys = y_prevs[:, 0:1]
            for i in range(0, y_prevs.size(1)):
                if is_train and random.random() < self.prob:
                    ys[:, [-1]] = y_prevs[:, [i]]
                
                x = self.decoder(ys, src_pad_mask, enc_outputs)
                out = self.final_fc(x)
        
                if self.classification is True:
                    ys = torch.cat((y_prevs[:, 0:1], torch.argmax(out, dim = -1)), dim = 1)
                else:
                    ys = torch.cat((y_prevs[:, 0:1], out), dim = 1)
    
            if self.classification is True:
                y_probs = torch.softmax(out, dim = -1)  # probability
                return y_probs, ys[:, 1:]
            else:
                return ys[:, 1:]
        

class MlTransformer(nn.Module):
    def __init__(self, enc_input_dim, dec_input_dim, out_dim, num_head, d_model, d_ff, N,
                 dropout=0., prob=0., is_classification=False, *args, **kwargs):
        """
        transformer used in machine learning
        :param enc_input_dim: refering to the encoder input feature dimension because there is no need to embedding
        :param dec_input_dim: refering to the decoder input feature dimension because there is no need to embedding
        :param out_dim: output or target feature dimension
        :param num_head: num of multi heads in attention
        :param d_model: normal feature dimension if most cases of Transformer
        :param d_ff: feature dimension in FeedForward Layer
        :param N: num of EncoderLayers or DecoderLayers
        :param dropout: dropout ration
        :param prob: the probability for step i+1 import ground truth of step i+1
        :param is_classification: classification if True else regression
        :param args:
        :param kwargs:
        """
        super(MlTransformer, self).__init__()
        self.model_conf = {
            'enc_input_dim': enc_input_dim,
            'dec_input_dim': dec_input_dim,
            'out_dim': out_dim,
            'num_head': num_head,
            'd_model': d_model,
            'd_ff': d_ff,
            'N': N,
            'dropout': dropout,
            'prob': prob,
            'classification': is_classification
        }

        self.grid_att_fc = nn.Linear(enc_input_dim, out_dim)
        self.enc_input_fc = nn.Linear(enc_input_dim + out_dim, d_model)
        self.encoder = Encoder(num_head, d_model, d_ff, N, dropout = dropout)
        self.dec_input_fc = nn.Linear(dec_input_dim + out_dim, d_model)
        self.decoder = Decoder(num_head, d_model, d_ff, N, dropout = dropout)
        self.final_fc = nn.Linear(d_model, out_dim)
        self.prob = prob
        self.classification = is_classification
        
        # these blocks blow are no need in machine leaning because of its uniform format
        # self.encoder.add_pe = None
        self.encoder.src_emb = None
        self.encoder.get_attn_pad_mask = None
        # self.decoder.add_pe = None
        self.decoder.tgt_emb = None
    
    def forward(self, x_hist, x_future, y_prevs, x_grid, is_train: bool):
        """
        in case of inference, y_prevs can be [batch_size, 1, tgt_len], due to the future target is unvailable.
        :param x_hist: [batch_size, src_len, src_size]
        :param x_future: decoding external input
        :param y_prevs: [batch_size, tgt_len, tgt_len] or [batch_size, 1, tgt_len]
        :param x_grid: (batch_size, src_len, n_grids, src_dim)
        :param is_train: train mode if is_train is True or test mode
        :return:
        """
        
        grid_atten_value, _ = attention(self.grid_att_fc(x_hist.unsqueeze(2)), x_grid, x_grid, atten_mask = None)
        enco_inputs = torch.cat((x_hist, grid_atten_value.squeeze(2)), dim = -1)
        enco_inputs = self.enc_input_fc(enco_inputs)
        
        # enc_outputs: (batch_size, src_len, d_model), src_pad_mask: (batch_size, src_len, src_len)
        enc_outputs, src_pad_mask = self.encoder(enco_inputs)
        
        out = torch.tensor([]).to(x_future.device)
        ys = y_prevs[:, 0:1]
        for i in range(0, x_future.size(1)):
            if is_train and random.random() < self.prob:
                ys[:, [-1]] = y_prevs[:, [i]]
            x = torch.cat((x_future[:, :i+1], ys), dim = -1)
            x = self.dec_input_fc(x)
            x = self.decoder(x, src_pad_mask, enc_outputs)
            out = self.final_fc(x)

            if self.classification is True:
                ys = torch.cat((y_prevs[:, 0:1], torch.argmax(out, dim = -1)), dim = 1)
            else:
                ys = torch.cat((y_prevs[:, 0:1], out), dim = 1)

        if self.classification is True:
            y_probs = torch.softmax(out, dim = -1)  # probability
            return y_probs, ys[:, 1:]
        else:
            return ys[:, 1:]


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
        
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
    
    def __len__(self):
        return self.enc_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


if __name__ == '__main__':
    import torchtext
    data_dir = '../../data/nlp/de2en'
    torchtext.datasets.Multi30k(root = data_dir, split = ('train', 'valid', 'test'), language_pair = ('de', 'en'))
    
    #  src_len = 5  # enc_input max sequence length, tgt_len = 7  # dec_input(=dec_output) max sequence length
    sentences = [
        # enc_input                dec_input            dec_output
        ['ich mochte ein bier P', 'S i want a beer . P', 'i want a beer . P E'],
        ['ich mochte zehr ein cola', 'S i really want a coke .', 'i really want a coke . E']
    ]
    
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5, 'zehr': 6}
    src_idx2word = {i: w for i, w in enumerate(src_vocab)}
    enco_seq_len = len(src_vocab)
    
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8, 'really': 9}
    tgt_idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    deco_seq_len = len(tgt_vocab)
    
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    model = Transformer(enco_seq_len, deco_seq_len, deco_seq_len, n_heads, d_model, d_ff, n_layers)
    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.99)
    
    for epoch in range(300):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            # enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            ys_prob, ys_cls = model(enc_inputs, dec_inputs, True, True)
            loss = criterion(ys_prob.reshape(-1, ys_prob.shape[-1]), dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # -------- 测试 -------
    enc_inputs, dec_inputs, _ = next(iter(loader))
    _, ys_cls = model(enc_inputs, dec_inputs, False, False)

    enc_inputs, ys_cls = enc_inputs.tolist(), ys_cls.tolist()
    for i in range(len(enc_inputs)):
        enc_inputs[i] = [src_idx2word[idx] for idx in enc_inputs[i]]
        ys_cls[i] = [tgt_idx2word[idx] for idx in ys_cls[i]]
    
    for enc, pred in zip(enc_inputs, ys_cls):
        print(enc, '->', pred)

