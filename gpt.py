import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 768
VOVAB_SIZE = 50_000

input_sentences = ['I like when robots do', 'When the time comes, you will know about this']



class RotaryPositionalEmbedding:
    # define it yourself
    pass

class TrainableEmbedding(x):
    pass


class GPT(nn.Module):
    def __init__(self):
        super().__init__()  
        
        # define a linear layer
        self.linear = nn.Linear(32, 32)
        
        # define a softmax layer
        self.softmax = nn.Softmax(dim=-1)
        
        # define attention layer
        self.attn = nn.MultiheadAttention(32, 8)

        #self.rotary = RotaryPositionalEmbedding()
        self.embedding = TrainableEmbedding()
        
        
    def forward(self, x):
        # apply rotary positional embedding
        #x = self.rotary(x)
        x = self.embedding(x)
        
        # apply linear layer
        x = self.linear(x)
        
        # apply softmax layer
        x = self.softmax(x)
        
        # apply attention layer
        x, _ = self.attn(x, x, x)
        
        # apply linear layer
        x = self.linear(x)
        
        return x
    

        
