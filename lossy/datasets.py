from copy import deepcopy

import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import TensorDataset

def get_dataset(dataset, data_path, batch_size=64, standardize=True, split_test_off_train=False,
                first_n=None, eval_batch_size=256):
    if dataset == 'MNIST':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        if not standardize:
            mean = [0]
            std = [1]
        else:
            mean = [0.1307]
            std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(32, ),  # transforms.InterpolationMode.BILINEAR),
                                        transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0))])
        dst_train = torchvision.datasets.MNIST(data_path, train=True, download=False, transform=transform) # no augmentation
        dst_test = torchvision.datasets.MNIST(data_path, train=False, download=False, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        if not standardize:
            mean = [0]
            std = [1]
        else:
            mean = [0.2861]
            std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(32, ),  # transforms.InterpolationMode.BILINEAR),
                                        transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0))])
        dst_train = torchvision.datasets.FashionMNIST(data_path, train=True, download=False, transform=transform) # no augmentation
        dst_test = torchvision.datasets.FashionMNIST(data_path, train=False, download=False, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        if not standardize:
            mean = [0., 0., 0.]
            std = [1, 1, 1]
        else:
            mean = [0.4377, 0.4438, 0.4728]
            std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = torchvision.datasets.SVHN(data_path, split='train', download=False, transform=transform)  # no augmentation
        dst_test = torchvision.datasets.SVHN(data_path, split='test', download=False, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        if not standardize:
            mean = [0., 0., 0.]
            std = [1, 1, 1]
        else:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = torchvision.datasets.CIFAR10(data_path, train=True, download=False, transform=transform) # no augmentation
        dst_test = torchvision.datasets.CIFAR10(data_path, train=False, download=False, transform=transform)
        class_names = dst_train.classes
    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        if not standardize:
            mean = [0., 0., 0.]
            std = [1, 1, 1]
        else:
            assert False, "need to check values"
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = torchvision.datasets.CIFAR100(data_path, train=True, download=False, transform=transform) # no augmentation
        dst_test = torchvision.datasets.CIFAR100(data_path, train=False, download=False, transform=transform)
        class_names = dst_train.classes
    elif dataset == 'USPS':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        assert not standardize
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(32, ),  # transforms.InterpolationMode.BILINEAR),
                                        transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0))])
        dst_train = torchvision.datasets.USPS(data_path, train=True, download=False, transform=transform)  # no augmentation
        dst_test = torchvision.datasets.USPS(data_path, train=False, download=False, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
    else:
        raise NotImplementedError('unknown dataset: %s'%dataset)

    if first_n is not None:
        dst_train = torch.utils.data.Subset(dst_train, np.arange(0,first_n))
        dst_test = torch.utils.data.Subset(dst_test, np.arange(0,first_n))
    if split_test_off_train:
        n_train = len(dst_train)
        n_split = int(np.ceil(n_train * 0.8))
        dst_test = torch.utils.data.Subset(deepcopy(dst_train), np.arange(n_split,n_train))
        dst_train = torch.utils.data.Subset(deepcopy(dst_train), np.arange(0, n_split))

    trainloader = torch.utils.data.DataLoader(dst_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True)
    train_det_loader = torch.utils.data.DataLoader(dst_train,
                                            batch_size=eval_batch_size,
                                            shuffle=False,
                                            num_workers=2,
                                            drop_last=False)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    return channel, im_size, num_classes, class_names, trainloader, train_det_loader, testloader


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_a, dataset_b, mix_x_func, ):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.mix_x_func = mix_x_func

    def __getitem__(self, index):
        X_a, y_a = self.dataset_a[index]
        i_random = np.random.choice(len(self.dataset_b), )
        # hack to try to enforce that everybody gets different images
        # in case somehow i_random all are same due to multithreading or whatever
        i_random = (i_random + index) % len(self.dataset_b)
        X_b, _ = self.dataset_b[i_random]
        X = self.mix_x_func(X_a, X_b)
        return X, y_a

    def __len__(self):
        return len(self.dataset_a)


def linear_interpolate_a_b(X_a, X_b, weight_a):
    X = X_a * weight_a + X_b * (1 - weight_a)
    return X


def multiply_X_a_b(X_a, X_b):
    return X_a * X_b



def restrict_to_classes(loader, i_classes, remap_labels):
    subset = restrict_dataset_to_classes(loader.dataset, i_classes, remap_labels=remap_labels)
    assert loader.sampler.__class__.__name__ in ['RandomSampler', 'SequentialSampler']
    shuffle = loader.sampler.__class__.__name__ == 'RandomSampler'
    return th.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


def restrict_dataset_to_classes(dataset, i_classes, remap_labels):
    dataset = deepcopy(dataset)
    found_key = None
    keys = ['train_labels', 'test_labels', 'targets', 'labels', 'tensors']
    for key in keys:
        if hasattr(dataset, key):
            found_key = key
            if key != 'tensors':
                labels = getattr(dataset, key)
            else:
                labels = dataset.tensors[1].argmax(dim=1).detach().cpu().numpy()
            indices = [np.flatnonzero(np.array(labels) == i_class) for i_class in i_classes]
            indices = np.sort(np.concatenate(indices))

    if found_key is None:
        assert hasattr(dataset, 'tensors')

    if remap_labels:
        resubtract = dict([(i_old_cls, i_old_cls - i_new_cls)
                           for i_new_cls, i_old_cls in
                           enumerate(i_classes)])
        labels = [l - resubtract[int(l)] if int(l) in resubtract else l for l in
                  labels]

    if found_key == 'tensor':
        new_labels = F.one_hot(
            th.tensor(labels), num_classes=dataset.tensors[1].shape[1]
        ).type_as(dataset.tensors[1])[indices].clone()
        subset = TensorDataset(dataset.tensors[0][indices].clone(),
                               new_labels)
    else:
        dataset.__dict__[found_key] = labels
        subset = th.utils.data.Subset(dataset, indices)
    return subset
