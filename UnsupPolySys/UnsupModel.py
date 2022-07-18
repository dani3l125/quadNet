import torch
import torch.nn as nn
import sys
import os.path

sys.path.append(os.path.join('..', 'PNPData'))

from PNPData.PNP import resize_vector


class UnsupModel(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.roots = nn.Parameter(data=torch.rand(n), requires_grad=True)

    def forward(self, system):
        return (system @ resize_vector(self.roots.unsqueeze(0)).T).squeeze()
