from torch import nn
import torch as th
import numpy as np
import kornia


class Augmenter(nn.Module):
    def __init__(self, contrast_factor, mix_grayscale_dists, demean,
                 expect_glow_range, std_noise_factor):
        super().__init__()
        self.contrast_factor = contrast_factor
        self.mix_grayscale_dists = mix_grayscale_dists
        self.demean = demean
        self.expect_glow_range = expect_glow_range
        self.std_noise_factor = std_noise_factor

    def forward(self, x):
        if self.expect_glow_range:
            x = x + 0.5
        if self.mix_grayscale_dists:
            x_flat = th.flatten(x, 1)
            x_other = th.flatten(x[th.randperm(len(x))], 1)
            x_val_sorted = x_other.sort(dim=1)[0]
            x_i_sorted = x_flat.sort(dim=1)[1].sort(dim=1)[1]
            changed_x = th.gather(x_val_sorted, 1, x_i_sorted, )
            changed_x = x_flat + (changed_x - x_flat).detach()
            reshaped_x = changed_x.reshape(x.shape)
            x = reshaped_x
        if self.demean:
            x = x - x.mean(dim=(1, 2, 3), keepdim=True) + 0.5
            # now clamp to [0,1]
            x = x + (x.clamp(1e-3,1-1e-3) - x).detach()
        contrast_factor = self.contrast_factor
        if callable(contrast_factor):
            contrast_factor = contrast_factor()

        x = kornia.enhance.adjust_contrast(x, contrast_factor)

        if self.std_noise_factor is not None:
            x = x + th.randn_like(x) * self.std_noise_factor
            # now clamp to [0,1]
            x = x + (x.clamp(1e-3,1-1e-3) - x).detach()

        if self.expect_glow_range:
            x = x - 0.5

        return x