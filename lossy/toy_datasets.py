from copy import deepcopy

import numpy as np
import torch
import torch as th
import kornia

from lossy.datasets import linear_interpolate_a_b
from lossy.util import np_to_th
from lossy import data_locations
from torchvision import transforms
from torchvision.datasets import ImageNet


# maybe can be imported from lossy.datasets instead? version is slightly different
# this oen seems mroe complete and newer
# However this somehow always matched exact same indices hmmhm
class MixedSet(th.utils.data.Dataset):
    def __init__(self, set_a, set_b, merge_function_x, merge_function_y):
        self.set_a = set_a
        self.set_b = set_b
        self.merge_function_x = merge_function_x
        self.merge_function_y = merge_function_y

    def __getitem__(self, index):
        x_a, y_a = self.set_a[index]
        x_b, y_b = self.set_b[index]

        x = self.merge_function_x(x_a, x_b)
        y = self.merge_function_y(y_a, y_b)
        return x, y

    def __len__(self):
        return min(len(self.set_a), len(self.set_b))


class StripesSet(th.utils.data.Dataset):
    def __init__(self, orig_set, label_from, stripes_factor, im_size, freq):
        self.orig_set = orig_set
        self.sin_vals = np.sin(np.linspace(0, freq * np.pi, im_size))
        self.sin_vals = self.sin_vals * 0.5 + 0.5
        # make proper range
        self.label_from = label_from
        self.stripes_factor = stripes_factor

    def __getitem__(self, index):
        x_orig, y_orig = self.orig_set[index]
        horizontal = th.rand(1).item() > 0.5
        n_sin_vals = len(self.sin_vals)
        if horizontal:
            x_orig.data[:, :, :] = (
                x_orig.data[:, :, :] * (1 - self.stripes_factor)
                + np_to_th(self.sin_vals, dtype=np.float32).reshape(1, n_sin_vals, 1) * self.stripes_factor
            ).data[:]
        else:

            x_orig.data[:, :, :] = (
                x_orig.data[:, :, :] * (1 -  self.stripes_factor)
                + np_to_th(self.sin_vals, dtype=np.float32).reshape(1, 1, n_sin_vals) * self.stripes_factor
            ).data[:]
        if self.label_from == "stripes":
            y = int(horizontal)
        elif self.label_from == "orig_set":
            y = y_orig
        return x_orig, y

    def __len__(self):
        return len(self.orig_set)


class AddUniformNoise(th.utils.data.Dataset):
    def __init__(self, orig_set, add_noise_factor=0.5):
        self.orig_set = orig_set
        self.add_noise_factor = add_noise_factor

    def __getitem__(self, index):
        x_orig, y_orig = self.orig_set[index]
        uniform_noise = th.rand_like(x_orig)
        uniform_noise = uniform_noise * x_orig
        merged_x = uniform_noise * self.add_noise_factor + x_orig * (1 - self.add_noise_factor)
        return merged_x, y_orig

    def __len__(self):
        return len(self.orig_set)


def load_dataset(
    dataset_name,
    data_path,
    reverse,
    first_n,
    split_test_off_train,
    batch_size,
    stripes_factor,
    eval_batch_size=256,
):
    assert dataset_name in [
        "mnist_fashion",
        "stripes",
        "mnist_cifar",
        "stripes_imagenet",
        "mnist_uniform",
    ]

    from lossy.datasets import get_train_test_datasets

    if dataset_name == "mnist_fashion":
        num_classes = 10
        train_mnist, test_mnist = get_train_test_datasets(
            "MNIST", data_path, standardize=False
        )
        train_fashion, test_fashion = get_train_test_datasets(
            "FashionMNIST", data_path, standardize=False
        )

        def merge_x_fn(x_a, x_b):
            if th.rand(1).item() > 0.5:
                x = th.cat((x_a, x_b), dim=2)
            else:
                x = th.cat((x_b, x_a), dim=2)

            x = kornia.resize(x, x_a.size()[1:], align_corners=False)
            return x

        def merge_y_fn(y_a, y_b):
            return y_a

        if not reverse:
            dst_train = MixedSet(train_mnist, train_fashion, merge_x_fn, merge_y_fn)
            dst_test = MixedSet(test_mnist, test_fashion, merge_x_fn, merge_y_fn)
        else:
            dst_train = MixedSet(train_fashion, train_mnist, merge_x_fn, merge_y_fn)
            dst_test = MixedSet(test_fashion, test_mnist, merge_x_fn, merge_y_fn)
    elif dataset_name == "mnist_uniform":
        train_mnist, test_mnist = get_train_test_datasets(
            "MNIST", data_path, standardize=False
        )
        num_classes = 10
        dst_train = AddUniformNoise(train_mnist, add_noise_factor=0.5)
        dst_test = AddUniformNoise(test_mnist, add_noise_factor=0.5)
    elif dataset_name == "stripes":

        train_cifar, test_cifar = get_train_test_datasets(
            "CIFAR10", data_path, standardize=False
        )
        label_from = ["stripes", "orig_set"][reverse]
        num_classes = [2, 10][reverse]
        dst_train = StripesSet(train_cifar, label_from=label_from, stripes_factor=stripes_factor, im_size=32, freq=11)
        dst_test = StripesSet(test_cifar, label_from=label_from, stripes_factor=stripes_factor, im_size=32, freq=11)
    elif dataset_name == "stripes_imagenet":

        train_imagenet, test_imagenet = get_train_test_datasets(
            "IMAGENET", data_path, standardize=False
        )
        label_from = ["stripes", "orig_set"][reverse]
        num_classes = [2, 1000][reverse]
        dst_train = StripesSet(train_imagenet, label_from=label_from, stripes_factor=stripes_factor, im_size=224, freq=11)
        dst_test = StripesSet(test_imagenet, label_from=label_from, stripes_factor=stripes_factor, im_size=224, freq=11)
    elif dataset_name == "mnist_cifar":
        num_classes = 10
        from lossy.datasets import MixedDataset
        import functools

        train_mnist, test_mnist = get_train_test_datasets(
            "MNIST", data_path, standardize=False
        )
        train_cifar, test_cifar = get_train_test_datasets(
            "CIFAR10", data_path, standardize=False
        )
        if not reverse:
            dst_train = MixedDataset(
                train_mnist,
                train_cifar,
                functools.partial(linear_interpolate_a_b, weight_a=0.5),
            )
            dst_test = MixedDataset(
                test_mnist,
                test_cifar,
                functools.partial(linear_interpolate_a_b, weight_a=0.5),
            )
        else:
            dst_train = MixedDataset(
                train_cifar,
                train_mnist,
                functools.partial(linear_interpolate_a_b, weight_a=0.5),
            )
            dst_test = MixedDataset(
                test_cifar,
                test_mnist,
                functools.partial(linear_interpolate_a_b, weight_a=0.5),
            )
    else:
        assert False

    if first_n is not None:
        dst_train = torch.utils.data.Subset(dst_train, np.arange(0, first_n))
        dst_test = torch.utils.data.Subset(dst_test, np.arange(0, first_n))
    if split_test_off_train:
        n_train = len(dst_train)
        n_split = int(np.ceil(n_train * 0.8))
        dst_test = torch.utils.data.Subset(
            deepcopy(dst_train), np.arange(n_split, n_train)
        )
        dst_train = torch.utils.data.Subset(deepcopy(dst_train), np.arange(0, n_split))

    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    train_det_loader = torch.utils.data.DataLoader(
        dst_train,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=eval_batch_size, shuffle=False, num_workers=2
    )

    return (
        num_classes,
        trainloader,
        train_det_loader,
        testloader,
    )
