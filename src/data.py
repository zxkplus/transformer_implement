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

from typing import Any, Sequence, List, Tuple
def pad_sequences(
    field1: Sequence[Any],
    field2: Sequence[Any],
    fill_value: Any = 0
) -> Tuple[List[Any], List[Any]]:
    """
    比较两个序列的长度，用 fill_value 在较短的序列右侧填充至等长。

    Args:
        field1: 任意序列（支持 len() 和迭代）
        field2: 任意序列
        fill_value: 用于填充的元素，默认为 0

    Returns:
        两个新列表，长度相同，较短的已填充 fill_value。
    """
    max_len = max(len(field1), len(field2))
    list1 = list(field1)
    list2 = list(field2)

    if len(list1) < max_len:
        list1.extend([fill_value] * (max_len - len(list1)))
    if len(list2) < max_len:
        list2.extend([fill_value] * (max_len - len(list2)))

    return list1, list2
def generate_addition_data(max_digits=3):
    x1 = random.randint(10**max(max_digits-2,0), 10**max(max_digits-1,0))
    x2 = random.randint(10**max(max_digits-2,0), 10**max(max_digits-1,0))
    r1 = x1 + x2
    ##补充一下如果不一样长就将位置对齐
    expression_str = str(x1) + "+" + str(x2)
    result_str = str(r1)
    return pad_sequences(encode(expression_str),encode(result_str,True),0)

def collate_batch(batch):
    datalist = []
    for _ in range(batch):
        datalist.append(generate_addition_data())
    return datalist