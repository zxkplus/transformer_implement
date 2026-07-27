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
    def __init__(self, d_model, num_heads):
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