# -*- coding: utf-8 -*-
"""
Created on 2022/1/2 下午4:24

@Project -> File: Autoformer -> fft.py

@Author: chenjun

@Describe:
"""

import numpy as np


def FFT(samples):
    """
    快速傅里叶变换计算各个基频率信号对应的权重系数F(k)
    使用二分和递归方案来提速，函数的参数f为采样点集合，长度为N
    当N=1时，使用G_k和H_k的定义来计算其值，否则，便将其分为偶序列和奇序列两个部分，
        使用递归的方式来进行求解
    k的取值范围为 range(0, N/2)，当前递归输出out的长度为N
    在一次递归过程中，假定不同k所对应的G_k和H_k两个序列已经获得，需要对不同的k进行遍历
        F(k) = G_k + W_N^k * H_k
        F(k + N/2) = G_k - W_N^k * H_k
    :param samples: 上一层分割的采样点集合
    """
    # 上一层分的采样点数目
    N = len(samples)
    
    # 如果集合中只剩1个采样点，那么利用W和H的定义来计算其值，即为采样点本身的值
    if N == 1:
        return samples
    
    # 按排列序号拆分为偶序列和奇序列两个部分
    SW = samples[0::2]
    SH = samples[1::2]
    
    # 分别调用FFT算法，计算得到W_k和H_k两个序列，长度与k的长度一致
    G = FFT(SW)
    H = FFT(SH)
    
    # K为需要计算的k的个数，out序列长度等于待求取的k的个数，也等于当前输入集合中采样点的个数
    K = int(N/2)
    out = np.empty(shape = N, dtype = np.complex64)
    
    # W_N^k，即为单位根W_N的k次幂，先定义出单位根
    w = np.cos(-2 * np.pi / N) + np.sin(-2 * np.pi / N) * 1j
    
    for k in range(0, K):
        out[k] = G[k] + w ** k * H[k]
        out[k + K] = G[k] - w ** k * H[k]
    return out


def IFFT(weights):
    """
    已知快速傅里叶变换各频率信号的系数权重，反向回求原信号采样值
    由于最终结果需要除以权重值的个数，因此需要在此基础上封装一层
    :param weights: 上一层分割的权重值集合
    """
    def _IFFT(weights):
        """
        递归函数，用以计算每次分割的G、H，最终的f(n)值
        """
        # 当序列只有一个样本时，使用G_n和H_n的定义来计算其值，即为对应权重值本身
        K = len(weights)  # 当前输入集合中权重值的个数
        if K == 1:
            return weights
        
        # 将序列分成奇偶两个序列，并分别递归求逆变换的结果
        SW = weights[0::2]
        SH = weights[1::2]
        G = _IFFT(SW)
        H = _IFFT(SH)
        
        # N为需要计算的n的个数，out序列长度等于待求取的n的个数，也等于当前输入集合中权重的个数
        N = int(K / 2)
        out = np.empty(shape = K, dtype = np.complex64)
        
        # W_N^n，即为单位根W_N的n次幂，先定义出单位根
        w = np.cos(2 * np.pi / K) + np.sin(2 * np.pi / K) * 1j
        for n in range(0, N):
            out[n] = (G[n] + w ** n * H[n])
            out[n + N] = (G[n] - w ** n * H[n])
        return out
    
    out = _IFFT(weights)
    out /= len(out)
    return out


if __name__ == '__main__':
    src = [8, 4, 5, 2, 1, 4, 5, 8]
    src = np.array(src, dtype = np.complex64)
    fft_res = FFT(src)
    ifft_res = IFFT(fft_res)
    np_fft_res = np.fft.fft(src)
    np_ifft_res = np.fft.ifft(np_fft_res)

    print(f'orgin input: {src} \n')
    print(f'fft result: {fft_res} \n')
    print(f'ifft result: {ifft_res} \n')
    print(f'fft result is correst: {np.allclose(fft_res, np_fft_res)}')
    print(f'ifft result is correst: {np.allclose(ifft_res, np_ifft_res)}')
