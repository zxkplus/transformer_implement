import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.line1 = nn.Linear(d_model,d_model * 4)
        self.line2 = nn.Linear(d_model * 4,d_model)
    def forward(self,x):
        return self.line2(torch.relu(self.line1(x)))