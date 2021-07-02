import numpy as np

def get_bpd(gen, X):
    n_dims = np.prod(X.shape[1:])
    _, lp = gen(X)
    bpd = -(lp - np.log(256) * n_dims) / (np.log(2) * n_dims)
    return bpd