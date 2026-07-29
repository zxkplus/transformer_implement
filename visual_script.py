from src.data import *
from src.transformer import *
from torch.optim import Adam
import torch.nn as nn
import torch
def predict(model,src_str,max_len=20):
    src_ids = encode(src_str)
    src_tensor = torch.tensor([src_ids]).to(device)
    tgt_ids = [1]
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids]).to(device)
        logits = model(src_tensor,tgt_tensor)
        next_id = logits[0, -1, :].argmax().item()
        if next_id == 2:  # <eos>
            break
        tgt_ids.append(next_id)
    return decode(tgt_ids[1:])

if __name__ == '__main__':
    vocab_size = 14
    num_steps = 5000
    batch_size = 64
    device = torch.device('cuda:0')
    model = Transformer(vocab_size,vocab_size,32,4,2).to(device)
    model.load_state_dict(torch.load('addition_model_2d.pt'))
    while True:
        expression = input("输入：")
        if expression == "Q": break
        result = predict(model,expression)
        print("结果 : ",result)