import torch
from torch import nn

from lossy.util import soft_clip, inverse_sigmoid


def quantize_data(x):
    x_0_255 = x * 255
    # pass-through grad
    x = (torch.round(x_0_255).detach() + x_0_255 - x_0_255.detach()) / 255.0
    return x


def to_plus_minus_one(x):
    return (x * 2) - 1


def add_glow_noise(x):
    # assume after conversion to glow range ([0,255/256.0])
    return x + torch.rand_like(x) * 1/256.0


def add_glow_noise_to_0_1(x):
    # later will be multiplied with 255/256.0
    return x + torch.rand_like(x) * 1/255.0


def glow_img_to_img_0_1(image):
    return (image + 0.5) * (256/255.0)


def img_0_1_to_glow_img(img_0_1):
    image_glow = ((img_0_1 * (255/256.0)) - 0.5)
    return image_glow


def img_0_1_to_cifar100_standardized(img):
    #https://github.com/chenyaofo/image-classification-codebase/blob/c199e524e32f79b2fcc6622734e78b4bcbbb5538/conf/cifar100.conf
    mean = [0.5070, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2761]
    std_th = torch.tensor(std, device='cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
    mean_th = torch.tensor(mean, device='cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
    normed = (img - mean_th) / std_th
    return normed


def img_0_1_to_cifar10_standardized(img):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    std_th = torch.tensor(std, device='cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
    mean_th = torch.tensor(mean, device='cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
    normed = (img - mean_th) / std_th
    return normed


def cifar10_standardized_to_img_0_1(img):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    std_th = torch.tensor(std, device='cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
    mean_th = torch.tensor(mean, device='cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
    unnormed = (img * std_th) + mean_th
    return unnormed


class ImageConverter(object):
    def __init__(self, image_standardize_before_glow, sigmoid_on_alpha,
                 standardize_for_clf, glow_noise_on_out, quantize_data):
        self.image_standardize_before_glow = image_standardize_before_glow
        self.sigmoid_on_alpha = sigmoid_on_alpha
        self.standardize_for_clf = standardize_for_clf
        self.glow_noise_on_out = glow_noise_on_out
        self.quantize_data = quantize_data
        if glow_noise_on_out:
            assert sigmoid_on_alpha
        if self.quantize_data:
            assert sigmoid_on_alpha

    def alpha_to_img_orig(self, alphas):
        if self.sigmoid_on_alpha:
            im_orig = torch.sigmoid(alphas)
            if self.quantize_data:
                # with grad
                im_orig = quantize_data(im_orig)
            if self.glow_noise_on_out:
                im_orig = im_orig + torch.rand_like(im_orig) * 1/255.0 # now onto [0,1+1/255.0=256/255.0], but later undone by to glow conversion
        else:
            if self.standardize_for_clf:
                im_orig = cifar10_standardized_to_img_0_1(alphas)
            else:
                im_orig = glow_img_to_img_0_1(alphas)
        return im_orig

    def img_orig_to_clf(self, img_orig):
        # img_orig should be in [0,1]
        #assert img_orig.min().item() >= 0, f"img_orig min was {img_orig.min().item()}"
        #assert img_orig.max().item() <= 1, f"img_orig nax was {img_orig.max().item()}"
        if self.standardize_for_clf:
            return img_0_1_to_cifar10_standardized(img_orig)
        else:
            return img_0_1_to_glow_img(img_orig)

    def alpha_to_clf(self, alphas):
        return self.img_orig_to_clf(self.alpha_to_img_orig(alphas))

    def alpha_to_glow(self, alphas):
        if self.image_standardize_before_glow and self.sigmoid_on_alpha:
            alphas = (alphas - alphas.mean(dim=(1, 2, 3), keepdim=True)) / (
                alphas.std(dim=(1, 2, 3), keepdim=True))
            alphas = alphas * 0.5

        img_glow = self.img_orig_to_glow(self.alpha_to_img_orig(alphas))
        # standardize to 0.15 std
        if self.image_standardize_before_glow and (not self.sigmoid_on_alpha):
            img_mean = img_glow.mean(dim=(1,2,3), keepdim=True)
            img_std = img_glow.std(dim=(1,2,3), keepdim=True)
            normed = (img_glow - img_mean) / img_std
            scaled = normed * 0.15
            remeaned = scaled + img_mean
            img_glow = remeaned
        return img_glow

    def img_orig_to_glow(self, img_orig):
        #assert img_orig.min().item() >= 0, f"img_orig min was f{img_orig.min().item()}"
        #assert img_orig.max().item() <= 1, f"img_orig nax was f{img_orig.max().item()}"
        return img_0_1_to_glow_img(img_orig)
    def img_glow_to_orig(self, img_glow):
        #assert img_orig.min().item() >= 0, f"img_orig min was f{img_orig.min().item()}"
        #assert img_orig.max().item() <= 1, f"img_orig nax was f{img_orig.max().item()}"
        return glow_img_to_img_0_1(img_glow)


def standardize_per_example(alphas, eps=0):
    dims = tuple(range(1,len(alphas.shape)))
    alphas = (alphas - alphas.mean(dim=dims, keepdim=True)) / (
                alphas.std(dim=dims, keepdim=True).clamp_min(eps))
    return alphas


def contrast_normalize(img_0_1, eps=1e-6):
        img_clipped = soft_clip(img_0_1, eps, 1-eps)
        alphas = inverse_sigmoid(img_clipped)
        alphas_standardized = standardize_per_example(alphas)
        img_normalized = torch.sigmoid(alphas_standardized)
        return img_normalized

class ContrastNormalize(nn.Module):
    def forward(self, img_0_1):
        return contrast_normalize(img_0_1, eps=1e-6)