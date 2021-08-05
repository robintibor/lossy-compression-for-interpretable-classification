import numpy as np
import torch as th
from torch import nn
from lossy.gaussian import get_gaussian_log_probs

from rtsutils.util import np_to_th


class HaarWavelet(nn.Module):
    def __init__(self, cat_at_end,):
        super(HaarWavelet, self).__init__()
        self.cat_at_end = cat_at_end
        self.diagonal_kernel = np_to_th([[1, -1], [-1, 1]],
                                         dtype=np.float32).unsqueeze(0).unsqueeze(1)

    def forward(self, x):
        self.diagonal_kernel = self.diagonal_kernel.to(x.device)
        pooled = th.nn.functional.avg_pool2d(x, (2, 2), stride=2, )
        assert pooled.shape[2] == x.shape[2] // 2
        assert pooled.shape[3] == x.shape[3] // 2

        pooled_half_horizontal = th.nn.functional.avg_pool2d(x, (1, 2), stride=(
        1, 2), )
        pooled_half_vertical = th.nn.functional.avg_pool2d(x, (2, 1),
                                                           stride=(2, 1), )

        diff_horizontal = pooled_half_horizontal[:, :,
                          0::2] - pooled_half_horizontal[:, :, 1::2]
        diff_vertical = pooled_half_vertical[:, :, :,
                        0::2] - pooled_half_vertical[:, :, :, 1::2]

        n_in_chans = x.shape[1]
        diff_diagonal = th.nn.functional.conv2d(
            x, self.diagonal_kernel.repeat(n_in_chans,1,1,1),
            stride=(2, 2),
            groups=n_in_chans)

        if self.cat_at_end:
            y = th.cat((pooled, diff_horizontal, diff_vertical, diff_diagonal),
                       dim=1)
        else:
            y = (pooled, diff_horizontal, diff_vertical, diff_diagonal)
        return y

    def invert(self, y):
        if self.cat_at_end:
            (pooled, diff_horizontal, diff_vertical, diff_diagonal) = th.chunk(
                y, 4, dim=1)
        else:
            (pooled, diff_horizontal, diff_vertical, diff_diagonal) = y
        self.diagonal_kernel = self.diagonal_kernel.to(pooled.device)

        inverted = pooled.repeat_interleave(2, dim=2).repeat_interleave(
            2, dim=3) * 4
        diff_hor_interleaved = diff_horizontal.repeat_interleave(2, dim=3) * 2
        inverted[:, :, 0::2] = inverted[:, :, 0::2] + diff_hor_interleaved
        inverted[:, :, 1::2] = inverted[:, :, 1::2] - diff_hor_interleaved
        diff_ver_interleaved = diff_vertical.repeat_interleave(2, dim=2) * 2
        inverted[:, :, :, 0::2] = inverted[:, :, :, 0::2] + diff_ver_interleaved
        inverted[:, :, :, 1::2] = inverted[:, :, :, 1::2] - diff_ver_interleaved
        diagonal_expanded = diff_diagonal.repeat_interleave(
            2, dim=2).repeat_interleave( 2, dim=3)
        diagonal_to_add = diagonal_expanded * self.diagonal_kernel.repeat(
            (1, 1, diff_diagonal.shape[2], diff_diagonal.shape[3]))
        inverted = inverted + diagonal_to_add
        inverted = inverted / 4
        return inverted


def get_all_wavelet_coeffs(x):
    all_coefs = []
    pooled = x
    while pooled.size()[-1] > 1:
        pooled, a,b,c = HaarWavelet(cat_at_end=False)(pooled)
        all_coefs.extend([a,b,c])
    all_coefs.append(pooled)
    return all_coefs


class WaveletAndColorDiff(nn.Module):
    def __init__(self, log_std):
        super().__init__()
        self.log_std = log_std

    def forward(self, x):
        coeffs = get_all_wavelet_coeffs(x)
        flat_coeffs = th.cat([th.flatten(c, 1) for c in coeffs], dim=1)
        wavelet_diffs = th.sum(flat_coeffs ** 2, dim=1)
        color_unmeaned = x - th.mean(x, dim=(1), keepdim=True)
        color_flat = th.flatten(color_unmeaned, start_dim=1)
        color_log_probs = get_gaussian_log_probs(
            th.zeros_like(color_flat[0]),
            self.log_std * th.ones_like(color_flat[0]),
            color_flat, )
        wavelet_log_probs = get_gaussian_log_probs(
            th.zeros_like(flat_coeffs[0]),
            self.log_std * th.ones_like(flat_coeffs[0]),
            flat_coeffs, )
        return x, wavelet_log_probs + color_log_probs