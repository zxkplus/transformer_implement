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
    vocab_size = 128
    num_steps = 5000
    batch_size = 64
    device = torch.device('cuda:0')
    model = Transformer(vocab_size,vocab_size,32,4,2).to(device)
    optimizer = Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)


    for step in range(num_steps):
        batch = collate_batch(batch_size)

        #转tensor
        src_token_list , tgt_token_list = zip(*batch)

        src_tensor = torch.tensor(src_token_list).to(device)           # 形状 (batch, src_len)
        tgt_tensor = torch.tensor(tgt_token_list).to(device)           # 形状 (batch, tgt_len)

        # 3. 构造 padding mask
        #    Transformer.forward 的 src_mask 参数接收 (batch, 1, 1, src_len)
        #    True = 遮住（不参与 attention）
        src_padding_mask = (src_tensor == 0).unsqueeze(1).unsqueeze(2)

        # 4. 切分出 decoder 输入和目标
        decoder_input = tgt_tensor[:, :-1]         # 去掉 <eos>
        target       = tgt_tensor[:, 1:]           # 去掉 <sos>

        # 5. 前向传播
        logits = model(src_tensor, decoder_input)

        # 6. 算 loss
        # reshape logits → (batch * seq, vocab_size)
        # reshape target  → (batch * seq,)
        loss = loss_fn(logits.reshape(-1, vocab_size), target.reshape(-1))

        # 7. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 8. 打印 loss
        if step % 100 == 0:
            print(f"step {step}, loss = {loss.item():.4f}")
    
    torch.save(model.state_dict(),'addition_model_4.pt')

    model.load_state_dict(torch.load('addition_model_4.pt'))
    ##开始实际预测
    experession_str = "2+146"
    result = predict(model,experession_str)
    print(experession_str,"=",result)