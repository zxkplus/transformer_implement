#词表
token_table = ['<pad>','<sos>','<eos>','0','1','2','3','4','5','6','7','8','9','+']
table2index = { {token_table[i]:i} for i in range(len(token_table))}
def encode(expr,add_sos_eos=False):
    tokenids = []
    for t in expr:
        tokenids.append(table2index[t])
    if add_sos_eos:
        tokenids = table2index[1] + tokenids + table2index[2]
    return tokenids

def decode(tokens):
    strtoken = ""
    for id in tokens:
        strtoken += token_table[id]
    return strtoken

def generate_addition_data(num_samples, max_digits=3):
    ...

def collate_batch(batch):
    ...