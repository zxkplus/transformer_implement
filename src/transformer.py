import torch
import torch.nn as nn
from src.encoder import TransformerEncoder
from src.decoder import TransformerDecoder
from src.positional import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,num_heads,num_layers,max_len = 5000):
        super().__init__()
        self.srcembedding = nn.Embedding(src_vocab_size,d_model)
        self.tgtembedding = nn.Embedding(tgt_vocab_size,d_model)
        self.causalmask = torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1)
        self.outlinear = nn.Linear(d_model,tgt_vocab_size)
        self.encoder = TransformerEncoder(d_model,num_heads,num_layers)
        self.decoder = TransformerDecoder(d_model,num_heads,num_layers)
        self.positional_embedding = PositionalEncoding(d_model,max_len)
    def forward(self,src,target):
        srcembedding = self.positional_embedding(self.srcembedding(src))
        tgtembedding = self.positional_embedding(self.tgtembedding(target))
        srcembedding = self.encoder(srcembedding)
        tgt_len = target.size(1)
        outvalue = self.outlinear(self.decoder(tgtembedding,srcembedding,self.causalmask[:tgt_len,:tgt_len]))
        return outvalue