import cv2
import numpy as np
from torchvision import transforms
import torch as th

from invglow.datasets import preprocess
from invglow.invertible.identity import Identity
from invglow.invertible.pure_model import NoLogDet


def modified_simclr_pipeline_transform(hflip, s=1, resized_crop=False, reflect_pad=True, ):
    # Compare https://github.com/sthalles/SimCLR/blob/e8a690ae4f4359528cfba6f270a9226e3733b7fa/data_aug/dataset_wrapper.py
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    if resized_crop:
        cropper = transforms.RandomResizedCrop(size=28),
    else:
        if reflect_pad:
            cropper = transforms.RandomCrop(size=32, padding=3, padding_mode='reflect')
        else:
            cropper = transforms.RandomAffine(0, translate=(0.1, 0.1))

    if hflip:
        flipper = transforms.RandomHorizontalFlip()
    else:
        flipper = NoLogDet(Identity())

    data_transforms = transforms.Compose([
        cropper,
        flipper,
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * 32)),
        transforms.ToTensor(),
        preprocess])
    return ReturnTwoTransformedExamples(data_transforms)


class ReturnTwoTransformedExamples(object):
    # Compare https://github.com/sthalles/SimCLR/blob/e8a690ae4f4359528cfba6f270a9226e3733b7fa/data_aug/dataset_wrapper.py
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


def compute_nt_xent_loss(za, zb, temperature=0.5):
    # compare https://github.com/sthalles/SimCLR/blob/e8a690ae4f4359528cfba6f270a9226e3733b7fa/loss/nt_xent.py
    z_a_b = th.cat((za, zb), dim=0)
    sim_all = th.nn.functional.cosine_similarity(z_a_b.unsqueeze(1), z_a_b.unsqueeze(0), dim=-1)
    l_pos = th.diag(sim_all, len(za))
    r_pos = th.diag(sim_all, -len(za))
    positives = th.cat([l_pos, r_pos]).view(2 * len(za), 1)
    diag = np.eye(2 * len(za))
    l1 = np.eye((2 * len(za)), 2 * len(za), k=-len(za))
    l2 = np.eye((2 * len(za)), 2 * len(za), k=len(za))
    mask = th.from_numpy((diag + l1 + l2))
    mask = (1 - mask).type(th.bool)
    negatives = sim_all[mask].view(2 * len(za), -1)

    logits = th.cat((positives, negatives), dim=1)
    logits = logits / temperature

    criterion = th.nn.CrossEntropyLoss(reduction="sum")
    labels = th.zeros(2 * len(za)).to(za.device).long()
    loss = criterion(logits, labels)
    return loss / (2 * len(za))


class ApplyMultipleTransforms:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        xs = [t(x) for t in self.transforms]
        return xs
