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

class EncoderBlock(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.multihead = MultiHeadAttention(d_model,num_heads)
        self.ffn = PositionwiseFeedForward(d_model)
        self.layernorm = torch.nn.LayerNorm(d_model)
    def forward(self,x):
        x1 = self.layernorm(x + self.multihead(x,x,x)[0])
        x2 = self.layernorm(x1 + self.ffn(x1))
        return x2

class TransformerEncoder(nn.Module):
    def __init__(self,d_model,num_heads,num_layers):
        super().__init__()
        self.encodesequence = nn.Sequential(*[EncoderBlock(d_model,num_heads) for i in range(num_layers)])
    def forward(self,x):
        return self.encodesequence(x)

class DecoderBlock(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)      # 自注意力
        self.cross_atten = MultiHeadAttention(d_model,num_heads)    # 交叉注意力
        self.ffn = PositionwiseFeedForward(d_model)                 # FFN
        self.norm1 = nn.LayerNorm(d_model)                          # 自注意力后的 norm
        self.norm2 = nn.LayerNorm(d_model)                          # 交叉注意力后的 norm
        self.norm3 = nn.LayerNorm(d_model)                          # FFN 后的 norm

    def forward(self,x,encode_output,mask=None):
        x1 = self.norm1(x + self.self_attn(x, x, x, mask=mask)[0])
        x2 = self.norm2(x1 + self.cross_atten(x1,encode_output,encode_output)[0])
        x3 = self.norm3(x2 + self.ffn(x2))
        return x3

class TransformerDecoder(nn.Module):
    def __init__(self,d_model,num_heads,num_layers):
        super().__init__()
        self.decodelist = nn.ModuleList([DecoderBlock(d_model,num_heads) for i in range(num_layers)])
    def forward(self,x,encode_output,mask=None):
        for layer in self.decodelist:
            x = layer(x,encode_output,mask)
        return x