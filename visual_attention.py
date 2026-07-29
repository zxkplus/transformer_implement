from src.data import *
from src.transformer import *
from torch.optim import Adam
import torch.nn as nn
import torch
from src.attention import MultiHeadAttention
import matplotlib.pyplot as plt

attention_data = {}

def make_hook(name):
    def hook(module,input,output):
        attention_data[name] = output[1].detach().cpu()
    return hook

if __name__ == '__main__':
    vocab_size = 14
    num_steps = 5000
    batch_size = 64
    device = torch.device('cuda:0')
    model = Transformer(vocab_size,vocab_size,32,4,2).to(device)
    batch = collate_batch(batch_size)

    #转tensor
    src_token_list , tgt_token_list = zip(*batch)

    src_tensor = torch.tensor(src_token_list).to(device)           # 形状 (batch, src_len)
    tgt_tensor = torch.tensor(tgt_token_list).to(device)           # 形状 (batch, tgt_len)
    model.load_state_dict(torch.load('addition_model_2d.pt'))
    # 3. 注册 hook
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            module.register_forward_hook(make_hook(name))

    _ = model(src_tensor,tgt_tensor)

    atten = attention_data['encoder.encodesequence.0.multihead']
    print(atten)
