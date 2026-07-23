import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    s = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s + mask
    attn = torch.softmax(s,-1)
    o = torch.matmul(attn,v)
    return o , attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads,):
        super().__init__()
        self.d_model  = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Qline = nn.Linear(d_model,d_model)
        self.Kline = nn.Linear(d_model,d_model)
        self.Vline = nn.Linear(d_model,d_model)
        self.outline = nn.Linear(d_model,d_model)
    
    def forward(self, q, k, v, mask=None):
        q = self.Qline(q)
        k = self.Kline(k)
        v = self.Vline(v)
        ##拆头
        B , T_q , _ = q.shape
        _ , T_k , _ = k.shape
        _ , T_v , _ = v.shape
        q = q.view(B,T_q, self.num_heads, self.d_k).transpose(1,2)
        k = k.view(B,T_k, self.num_heads, self.d_k).transpose(1,2)
        v = v.view(B,T_v, self.num_heads, self.d_k).transpose(1,2)
        out , attn = scaled_dot_product_attention(q,k,v,mask)
        out = out.transpose(1,2)
        out = out.reshape(B,T_q,self.d_model)
        out = self.outline(out)
        return out,attn
    
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
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.line1 = nn.Linear(d_model,d_model * 4)
        self.line2 = nn.Linear(d_model * 4,d_model)
    def forward(self,x):
        return self.line2(torch.relu(self.line1(x)))

        
    