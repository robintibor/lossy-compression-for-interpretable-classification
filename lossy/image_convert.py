import torch

def glow_img_to_img_0_1(image):
    return (image + 0.5) * (256/255.0)


def img_0_1_to_glow_img(img_0_1):
    image_glow = ((img_0_1 * (255/256.0)) - 0.5)#could add + (1/256*2)?robintibor@gmail.com or add at eval?
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
                 standardize_for_clf, glow_noise_on_out):
        self.image_standardize_before_glow = image_standardize_before_glow
        self.sigmoid_on_alpha = sigmoid_on_alpha
        self.standardize_for_clf = standardize_for_clf
        self.glow_noise_on_out = glow_noise_on_out
        if glow_noise_on_out:
            assert sigmoid_on_alpha


    def alpha_to_img_orig(self, alphas):
        if self.sigmoid_on_alpha:
            im_orig = torch.sigmoid(alphas)
            if self.glow_noise_on_out:
                im_orig = im_orig + torch.rand_like(im_orig) * 1/255.0
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


def standardize_per_example(alphas):
    dims = tuple(range(1,len(alphas.shape)))
    alphas = (alphas - alphas.mean(dim=dims, keepdim=True)) / (
                alphas.std(dim=dims, keepdim=True))
    return alphas