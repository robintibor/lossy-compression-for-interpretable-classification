import torch
import torchvision
from torchvision import transforms
import numpy as np

def get_dataset(dataset, data_path, batch_size=64, standardize=True, split_test_off_train=False,
                first_n=None):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        if not standardize:
            mean = [0]
            std = [1]
        else:
            mean = [0.1307]
            std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=transform)
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
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = torchvision.datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = torchvision.datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
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
        dst_train = torchvision.datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = torchvision.datasets.SVHN(data_path, split='test', download=True, transform=transform)
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
        dst_train = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
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
        dst_train = torchvision.datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = torchvision.datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    else:
        exit('unknown dataset: %s'%dataset)

    if first_n is not None:
        dst_train = torch.utils.data.Subset(dst_train, np.arange(0,first_n))
        dst_test = torch.utils.data.Subset(dst_test, np.arange(0,first_n))
    if split_test_off_train:
        n_train = len(dst_train)
        n_split = int(np.ceil(n_train * 0.8))
        dst_test = torch.utils.data.Subset(dst_train, np.arange(n_split,n_train))
        dst_train = torch.utils.data.Subset(dst_train, np.arange(0, n_split))

    trainloader = torch.utils.data.DataLoader(dst_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            drop_last=True)
    train_det_loader = torch.utils.data.DataLoader(dst_train,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=2,
                                            drop_last=False)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=2)
    return channel, im_size, num_classes, class_names, trainloader, train_det_loader, testloader
