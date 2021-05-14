import torch as th
from torch import nn


class AffineOnChans(nn.Module):
    def __init__(self, n_chans):
        super().__init__()
        self.adds = nn.Parameter(th.zeros(n_chans))
        self.factors = nn.Parameter(th.ones(n_chans))

    def forward(self, x):
        adds = self.adds.unsqueeze(0)
        factors = self.factors.unsqueeze(0)
        while len(adds.shape) < len(x.shape):
            adds = adds.unsqueeze(-1)
            factors = factors.unsqueeze(-1)
        z = (x + adds) * factors
        return z
