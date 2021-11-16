import torch as th
from torch import nn


def cos_sim_neg_grads(g_orig, g_simple):
    mask = g_orig < 0
    cos_sim = th.nn.functional.cosine_similarity(
            (mask * g_orig).flatten(1), (mask * g_simple).flatten(1), dim=1,
            eps=1e-12)
    return cos_sim

# def cos_sim_grads(g_orig, g_simple):
#     cos_sim = th.nn.functional.cosine_similarity(
#         g_orig.flatten(1), g_simple.flatten(1), dim=1,
#             eps=1e-12)
#     return cos_sim

def mse_neg_grads(g_orig, g_simple):
    mask = g_orig < 0
    diff  = (mask * g_orig).flatten(1) - (mask * g_simple).flatten(1)
    mse = th.mean(diff *diff, dim=1,)
    return mse


def indirect_change_strength(grads_a, grads_b):
    # assumed grads are lists (one element per parameter tensor),
    # each element then examples x params (of that tensor, already flattened)
    grads_a_flat = th.cat(grads_a, dim=1)
    grads_b_flat = th.cat(grads_b, dim=1)
    grad_alignment = th.sum(grads_a_flat * grads_b_flat, dim=1) / th.sum(
        grads_a_flat * grads_a_flat, dim=1
    )
    return th.mean(grad_alignment)


def grad_correlation(grads_a, grads_b, eps=1e-7):
    # assumed grads are lists (one element per parameter tensor),
    # each element then examples x params (of that tensor, already flattened)
    grads_a_flat = th.cat(grads_a, dim=1)
    grads_b_flat = th.cat(grads_b, dim=1)
    grads_corr = th.sum(grads_a_flat * grads_b_flat, dim=1) / (
        th.norm(grads_a_flat, dim=1, p=2) * th.norm(grads_b_flat, dim=1, p=2) + eps
    )
    return th.mean(grads_corr)


def grad_dot(grads_a, grads_b, eps=1e-7):
    grads_a_flat = th.cat(grads_a, dim=1)
    grads_b_flat = th.cat(grads_b, dim=1)
    grads_dot = th.sum(grads_a_flat * grads_b_flat, dim=1)
    return th.mean(grads_dot)


class RelevantLoss(nn.Module):
    def forward(self, out,y):
        return relevant_loss(out, y)


def relevant_loss(out, y):
    relevant_outs = th.stack([o[i_y] for o, i_y in zip(out, y)])
    return th.sum(relevant_outs)


class ClassOuts(nn.Module):
    def __init__(self, i_class):
        super().__init__()
        self.i_class = i_class

    def forward(self, out,y):
        relevant_outs = out[:,self.i_class]
        return th.sum(relevant_outs)