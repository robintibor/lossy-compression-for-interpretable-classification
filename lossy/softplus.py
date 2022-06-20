import torch as th
from torch import nn


class ReLUSoftPlusGrad(nn.Module):
    def __init__(self, softplus_mod):
        super().__init__()
        self.softplus = softplus_mod

    def forward(self, x):
        relu_x = th.nn.functional.relu(x)
        softplus_x = self.softplus(x)
        return softplus_x + (relu_x - softplus_x).detach()
