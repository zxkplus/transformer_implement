#词表
import random

token_table = ['<pad>','<sos>','<eos>','0','1','2','3','4','5','6','7','8','9','+']
table2index = { token_table[i]:i for i in range(len(token_table))}
def encode(expr,add_sos_eos=False):
    tokenids = []
    for t in expr:
        tokenids.append(table2index[t])
    if add_sos_eos:
        #tokenids = token_table[1] + tokenids + token_table[2]
        tokenids.insert(0,1)
        tokenids.append(2)
    return tokenids

def decode(tokens):
    strtoken = ""
    for id in tokens:
        strtoken += token_table[id]
    return strtoken
def generate_addition_data(min_digits = 1,max_digits=3):
    d1 = random.randint(min_digits, max_digits)
    d2 = random.randint(min_digits, max_digits)
    x1 = random.randint(10**(d1-1), 10**d1 - 1)
    x2 = random.randint(10**(d2-1), 10**d2 - 1)
    r1 = x1 + x2
    ##补充一下如果不一样长就将位置对齐
    expression_str = str(x1) + "+" + str(x2)
    result_str = str(r1)
    return encode(expression_str),encode(result_str,True)

def collate_batch(batch_size):
    """Generate a batch of addition samples and pad to uniform lengths."""
    datalist = []
    max_src_len = 0
    max_tgt_len = 0
    for _ in range(batch_size):
        src, tgt = generate_addition_data()
        max_src_len = max(max_src_len, len(src))
        max_tgt_len = max(max_tgt_len, len(tgt))
        datalist.append((src, tgt))

    for src, tgt in datalist:
        pad_src = max_src_len - len(src)
        pad_tgt = max_tgt_len - len(tgt)
        if pad_src > 0:
            src.extend([0] * pad_src)
        if pad_tgt > 0:
            tgt.extend([0] * pad_tgt)
    return datalist