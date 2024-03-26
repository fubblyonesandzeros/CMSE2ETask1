import torch
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        
    def forward(self, x1, x2, label):
        pennalise_same = label * (1-self.cos(x1, x2).abs())
        pennalise_not_same = (1-label) * self.cos(x1, x2).abs()
        loss = pennalise_same + pennalise_not_same
        return loss.mean()