import torch.nn as nn
import torch.nn.functional as F
from src.attention import MultiHeadAttention
from src.ffn import PositionwiseFeedForward

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