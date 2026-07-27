import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.attention import MultiHeadAttention
from src.ffn import PositionwiseFeedForward

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