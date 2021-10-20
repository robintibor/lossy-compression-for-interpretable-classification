import torch as th
import numpy as np
import random
import torch
from warnings import warn

def np_to_th(
    X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
    """
    Convenience function to transform numpy array to `torch.Tensor`.

    Converts `X` to ndarray using asarray if necessary.

    Parameters
    ----------
    X: ndarray or list or number
        Input arrays
    requires_grad: bool
        passed on to Variable constructor
    dtype: numpy dtype, optional
    var_kwargs:
        passed on to Variable constructor

    Returns
    -------
    var: `torch.Tensor`
    """
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor


def th_to_np(var):
    """Convenience function to transform `torch.Tensor` to numpy
    array.

    Should work both for CPU and GPU."""
    return var.cpu().data.numpy()

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


def soft_clip(x, vmin, vmax):
    return x + (x.clamp(vmin, vmax) -x).detach()


def set_random_seeds(seed, cuda):
    """Set seeds for python random module numpy.random and torch.

    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def np_to_var(
    X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
):
    warn("np_to_var has been renamed np_to_th, please use np_to_th instead")
    return np_to_th(
        X, requires_grad=requires_grad, dtype=dtype, pin_memory=pin_memory,
        **tensor_kwargs
    )

