import torch as th
from rtsutils.util import np_to_th
import numpy as np


def weighted_sum(total_weight, *args):
    weights = list(args)[::2]
    terms = list(args)[1::2]
    assert len(weights) == len(terms)
    weights = np_to_th(weights, dtype=np.float32, device=terms[0].device)
    terms = th.stack(terms)
    weights = weights * total_weight / th.sum(weights)
    return th.sum(terms * weights)


def inverse_sigmoid(x):
    return x.log() - (1-x).log()
