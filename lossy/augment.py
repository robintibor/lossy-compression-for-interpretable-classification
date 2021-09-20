import kornia
import torch as th
from torch import nn
import math
import torch
from torch.distributions.half_normal import HalfNormal
from enum import Enum
from torch import Tensor
from typing import List, Tuple, Optional, Dict
from torchvision.transforms import functional as F


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
):
    # from torchvision
    if op_name == "ShearX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(magnitude), 0.0],
        )
    elif op_name == "ShearY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(magnitude)],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            shear=[0.0, 0.0],
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            shear=[0.0, 0.0],
        )
    elif op_name == "Rotate":
        img = F.rotate(
            img,
            magnitude,
        )
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class TrivialAugmentPerImage(nn.Module):
    def __init__(self, n_examples, num_magnitude_bins, std_aug_magnitude):
        super().__init__()

        # This whole logic assumes the identity
        # is the smallest magnitude of augmentation
        if std_aug_magnitude is not None:
            dist = HalfNormal(num_magnitude_bins * std_aug_magnitude)
            cdfs = dist.cdf(th.linspace(1, num_magnitude_bins, num_magnitude_bins))
            # do not have to sum to zero as multinomial will normalize
            probs = th.cat((cdfs[0:1], th.diff(cdfs)))
        else:
            # Uniform sampling
            probs = torch.tensor([1 / num_magnitude_bins]).repeat(num_magnitude_bins)

        self.op_names = []
        self.magnitudes = []
        for i_example in range(n_examples):
            op_meta = self._augmentation_space(num_magnitude_bins)
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            i_magnitude_bin = torch.multinomial(probs, 1).item()
            magnitude = (
                float(magnitudes[i_magnitude_bin].item())
                if magnitudes.ndim > 0
                else 0.0
            )
            if signed and torch.randint(2, (1,)).item():
                magnitude *= -1.0
            self.op_names.append(op_name)
            self.magnitudes.append(magnitude)

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            # also seems not work for float images
            # "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            # grad problems
            # "AutoContrast": (torch.tensor(0.0), False),
            # "Equalize": (torch.tensor(0.0), False), not working for float images
        }

    def forward(self, X):
        assert len(X) == len(self.op_names) == len(self.magnitudes)
        aug_Xs = []
        for i_image, (op_name, magnitude) in enumerate(
            zip(self.op_names, self.magnitudes)
        ):
            aug_X = _apply_op(X[i_image : i_image + 1], op_name, magnitude)
            aug_Xs.append(aug_X)
        aug_X = th.cat(aug_Xs)
        assert len(aug_X) == len(X)
        return aug_X


class FixedAugment(nn.Module):
    def __init__(self, kornia_obj, X_shape):
        super().__init__()
        self.kornia_obj = kornia_obj
        # fix bug with forward parameters for padding
        if hasattr(kornia_obj, "padding"):
            assert kornia_obj.__class__.__name__ == "RandomCrop"
            padding = kornia_obj.padding
            X_shape = X_shape[:2] + (X_shape[2] + padding, X_shape[3] + padding)
        self.forward_parameters = self.kornia_obj.forward_parameters(X_shape)

    def forward(self, X):
        return self.kornia_obj(X, self.forward_parameters)


class Augmenter(nn.Module):
    def __init__(
        self,
        contrast_factor,
        mix_grayscale_dists,
        demean,
        expect_glow_range,
        std_noise_factor,
    ):
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
            changed_x = th.gather(
                x_val_sorted,
                1,
                x_i_sorted,
            )
            changed_x = x_flat + (changed_x - x_flat).detach()
            reshaped_x = changed_x.reshape(x.shape)
            x = reshaped_x
        if self.demean:
            x = x - x.mean(dim=(1, 2, 3), keepdim=True) + 0.5
            # now clamp to [0,1]
            x = x + (x.clamp(1e-3, 1 - 1e-3) - x).detach()
        contrast_factor = self.contrast_factor
        if callable(contrast_factor):
            contrast_factor = contrast_factor()

        x = kornia.enhance.adjust_contrast(x, contrast_factor)

        if self.std_noise_factor is not None:
            x = x + th.randn_like(x) * self.std_noise_factor
            # now clamp to [0,1]
            x = x + (x.clamp(1e-3, 1 - 1e-3) - x).detach()

        if self.expect_glow_range:
            x = x - 0.5

        return x
