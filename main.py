from src.data import *
from src.transformer import *
from torch.optim import Adam
import torch.nn as nn

if __name__ == '__main__':
    vocab_size = 13
    num_steps = 500
    batch_size = 16

    model = Transformer(vocab_size,vocab_size,32,4,2)
    optimizer = Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)


    for step in range(num_steps):
        batch = collate_batch(batch_size)

        #转tensor
        src_token_list , tgt_token_list = zip(*batch)

        src_tensor = torch.tensor(src_token_list)           # 形状 (batch, src_len)
        tgt_tensor = torch.tensor(tgt_token_list)           # 形状 (batch, tgt_len)

        # 3. 构造 padding mask
        #    Transformer.forward 的 src_mask 参数接收 (batch, 1, 1, src_len)
        #    True = 遮住（不参与 attention）
        src_padding_mask = (src_tensor == 0).unsqueeze(1).unsqueeze(2)

        # 4. 切分出 decoder 输入和目标
        decoder_input = tgt_tensor[:, :-1]         # 去掉 <eos>
        target       = tgt_tensor[:, 1:]           # 去掉 <sos>

        # 5. 前向传播
        logits = model(src_tensor, decoder_input, src_mask=src_padding_mask)

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