import torch as th


def indirect_change_strength(grads_a, grads_b):
    # assumed grads are lists (one element per parameter tensor),
    # each element then examples x params (of that tensor, already flattened)
    grads_a_flat = th.cat(grads_a, dim=1)
    grads_b_flat = th.cat(grads_b, dim=1)
    grad_alignment = th.sum(grads_a_flat * grads_b_flat, dim=1) / th.sum(
        grads_a_flat * grads_a_flat, dim=1
    )
    return th.mean(grad_alignment)


def grad_correlation(grads_a, grads_b):
    # assumed grads are lists (one element per parameter tensor),
    # each element then examples x params (of that tensor, already flattened)
    grads_a_flat = th.cat(grads_a, dim=1)
    grads_b_flat = th.cat(grads_b, dim=1)
    grads_corr = th.sum(grads_a_flat * grads_b_flat, dim=1) / (
        th.norm(grads_a_flat, dim=1, p=2) * th.norm(grads_b_flat, dim=1, p=2)
    )
    return th.mean(grads_corr)
