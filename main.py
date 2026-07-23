import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    s = torch.matmul(q,k.transpose(-1,-2)) / torch.sqrt(torch.tensor(q.shape[-1]))
    if mask is not None:
        s = s + mask
    attn = torch.softmax(s,-1)
    o = torch.matmul(attn,v)
    return o , attn