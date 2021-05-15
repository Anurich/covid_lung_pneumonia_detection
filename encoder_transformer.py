import torch
import torch.nn as nn
import torch.functional  as F
import numpy as np
from torchsummary import summary
from einops import rearrange

class multiheaded_attention(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.d_model = d_model
        self.head    = head
        # parallel attentions
        self.parallel_head = self.d_model//head
        print(self.parallel_head)
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        # Linear layer
        self.linear  = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, key, query, value):
        pass

    def forward(self,x):
        print(self.query(x).view(32, -1, self.head, self.parallel_head).transpose(1,2).shape)
        print(rearrange(self.query(x),'b n (h d)->b h n d', h = self.head).shape)
