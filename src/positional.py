import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len = 50000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        range_pos = torch.arange(0,max_len).unsqueeze(-1)
        freq_i = torch.arange(0,d_model // 2).unsqueeze(0)
        pe = torch.zeros((max_len,d_model))
        angle = range_pos / (10000 ** (freq_i * 2/ self.d_model))
        pe[:,0::2] = torch.sin(angle)
        pe[:,1::2] = torch.cos(angle)
        self.register_buffer("pe",pe)
    def forward(self,x):
        return x + self.pe[:x.size(1)]
