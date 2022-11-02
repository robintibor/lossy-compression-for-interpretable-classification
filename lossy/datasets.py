from copy import deepcopy

import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from lossy.mimic_cxr import MIMIC_CXR_JPG
from lossy import data_locations

from torchvision.datasets.imagenet import *


class ImageNet(ImageFolder):
    #https://github.com/automl/metassl/blob/cd686842cfd7db5bd506de566c3c2e8c2205bede/metassl/utils/imagenet.py
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.
    Copied from torchvision, besides warning below.
    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
        WARN::
        This is the same ImageNet class as in torchvision.datasets.imagenet, but it has the
        `ignore_archive` argument.
        This allows us to only copy the unzipped files before training.
    """

    def __init__(self, root, split="train", download=None, ignore_archive=False, **kwargs):
        if download is True:
            msg = (
                "The dataset is no longer publicly accessible. You need to "
                "download the archives externally and place them in the root "
                "directory."
            )
            raise RuntimeError(msg)
        elif download is False:
            msg = (
                "The use of the download flag is deprecated, since the dataset "
                "is no longer publicly accessible."
            )
            warnings.warn(msg, RuntimeWarning)

        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        if not ignore_archive:
            self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    def parse_archives(self):
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == "train":
                parse_train_archive(self.root)
            elif self.split == "val":
                parse_val_archive(self.root)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def get_dataset(
    dataset,
    data_path,
    batch_size=64,
    standardize=True,
    split_test_off_train=False,
    first_n=None,
    eval_batch_size=256,
    i_classes=None,
    mimic_cxr_clip=1.0,  # Clip MIMIC CXR Brightness?
    mimic_cxr_target=None,
    reverse=False,
    stripes_factor=0.15,
):
    if dataset.lower() in [
        "mnist_fashion",
        "stripes",
        "mnist_cifar",
    ]:
        from lossy.toy_datasets import load_dataset
        (
            num_classes,
            trainloader,
            train_det_loader,
            testloader,
        ) = load_dataset(
            dataset_name=dataset.lower(),
            data_path=data_path,
            reverse=reverse,
            first_n=first_n,
            split_test_off_train=split_test_off_train,
            batch_size=batch_size,
            stripes_factor=stripes_factor,
        )
        channel = 3
        im_size = (32, 32)
        class_names = None # ignore
        return (
            channel,
            im_size,
            num_classes,
            class_names,
            trainloader,
            train_det_loader,
            testloader,
        )
    assert not standardize
    if dataset == "MNIST":
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        if not standardize:
            mean = [0]
            std = [1]
        else:
            mean = [0.1307]
            std = [0.3081]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.Resize(
                    32,
                ),  # transforms.InterpolationMode.BILINEAR),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
            ]
        )
        dst_train = torchvision.datasets.MNIST(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = torchvision.datasets.MNIST(
            data_path, train=False, download=True, transform=transform
        )
        class_names = [str(c) for c in range(num_classes)]

    elif (dataset == "FashionMNIST") or (dataset == "FASHIONMNIST"):
        channel = 3
        im_size = (28, 28)
        num_classes = 10
        if not standardize:
            mean = [0]
            std = [1]
        else:
            mean = [0.2861]
            std = [0.3530]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.Resize(
                    32,
                ),  # transforms.InterpolationMode.BILINEAR),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
            ]
        )
        dst_train = torchvision.datasets.FashionMNIST(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = torchvision.datasets.FashionMNIST(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes

    elif dataset == "SVHN":
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        if not standardize:
            mean = [0.0, 0.0, 0.0]
            std = [1, 1, 1]
        else:
            mean = [0.4377, 0.4438, 0.4728]
            std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dst_train = torchvision.datasets.SVHN(
            data_path, split="train", download=True, transform=transform
        )  # no augmentation
        dst_test = torchvision.datasets.SVHN(
            data_path, split="test", download=True, transform=transform
        )
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == "CIFAR10":
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        if not standardize:
            mean = [0.0, 0.0, 0.0]
            std = [1, 1, 1]
        else:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dst_train = torchvision.datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = torchvision.datasets.CIFAR10(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes
    elif dataset == "CIFAR100":
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        if not standardize:
            mean = [0.0, 0.0, 0.0]
            std = [1, 1, 1]
        else:
            assert False, "need to check values"
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dst_train = torchvision.datasets.CIFAR100(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = torchvision.datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes
    elif dataset == "USPS":
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        assert not standardize
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    32,
                ),  # transforms.InterpolationMode.BILINEAR),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
            ]
        )
        dst_train = torchvision.datasets.USPS(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = torchvision.datasets.USPS(
            data_path, train=False, download=True, transform=transform
        )
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == "CELEBA":
        channel = 3
        im_size = (32, 32)
        num_classes = 40
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (32, 32),
                ),  # transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        dst_train = torchvision.datasets.CelebA(
            root=data_path,
            target_type="attr",
            download=True,
            transform=transform,
            split="train",
        )
        dst_test = torchvision.datasets.CelebA(
            root=data_path,
            target_type="attr",
            download=True,
            transform=transform,
            split="valid",
        )
        class_names = dst_train.attr_names
    elif dataset == "MIMIC-CXR":
        mimic_folder = data_locations.mimic_cxr
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.clamp_max(mimic_cxr_clip)),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
            ])
        n_dicoms = 60000
        dst_train = MIMIC_CXR_JPG(
            mimic_folder,
            mimic_cxr_target,
            "train",
            n_dicoms,
            transform=transform,
        )
        dst_test = MIMIC_CXR_JPG(
            mimic_folder,
            mimic_cxr_target,
            "validate",
            n_dicoms,
            transform=transform,
        )
        channel = 3
        im_size = (32, 32)
        class_names = dst_train.classes
        num_classes = len(class_names)
    elif dataset == 'IMAGENET':

        imagenet_root = data_locations.imagenet

        # https://github.com/pytorch/examples/blob/fcf8f9498e40863405fe367b9521269e03d7f521/imagenet/main.py#L213-L237
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # load the dataset
        dst_train = ImageNet(
            root=imagenet_root,
            split="train",
            transform=train_transform,
            ignore_archive=True,
        )

        dst_test = ImageNet(
            root=imagenet_root,
            split="val",
            transform=valid_transform,
            ignore_archive=True,
        )

        channel = 3
        im_size = (224, 224)
        class_names = dst_train.classes
        num_classes = len(class_names)
    else:
        raise NotImplementedError("unknown dataset: %s" % dataset)

    if i_classes is not None:
        dst_train = restrict_dataset_to_classes(dst_train, i_classes, remap_labels=True)
        dst_test = restrict_dataset_to_classes(dst_test, i_classes, remap_labels=True)
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
        channel,
        im_size,
        num_classes,
        class_names,
        trainloader,
        train_det_loader,
        testloader,
    )


def get_train_test_datasets(dataset, data_path, standardize=False):
    # For compatibility for prev code
    assert standardize is False
    (
        channel,
        im_size,
        num_classes,
        class_names,
        trainloader,
        train_det_loader,
        testloader,
    ) = get_dataset(dataset,data_path,standardize=standardize)
    return trainloader.dataset, testloader.dataset


class MixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_a,
        dataset_b,
        mix_x_func,
    ):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.mix_x_func = mix_x_func

    def __getitem__(self, index):
        X_a, y_a = self.dataset_a[index]
        i_random = np.random.choice(
            len(self.dataset_b),
        )
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
    subset = restrict_dataset_to_classes(
        loader.dataset, i_classes, remap_labels=remap_labels
    )
    assert loader.sampler.__class__.__name__ in ["RandomSampler", "SequentialSampler"]
    shuffle = loader.sampler.__class__.__name__ == "RandomSampler"
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
    keys = ["train_labels", "test_labels", "targets", "labels", "tensors"]
    for key in keys:
        if hasattr(dataset, key):
            found_key = key
            if key != "tensors":
                labels = getattr(dataset, key)
            else:
                labels = dataset.tensors[1].argmax(dim=1).detach().cpu().numpy()
            indices = [
                np.flatnonzero(np.array(labels) == i_class) for i_class in i_classes
            ]
            indices = np.sort(np.concatenate(indices))

    if found_key is None:
        assert hasattr(dataset, "tensors")

    if remap_labels:
        resubtract = dict(
            [
                (i_old_cls, i_old_cls - i_new_cls)
                for i_new_cls, i_old_cls in enumerate(i_classes)
            ]
        )
        labels = [l - resubtract[int(l)] if int(l) in resubtract else l for l in labels]

    if found_key == "tensor":
        new_labels = (
            F.one_hot(th.tensor(labels), num_classes=dataset.tensors[1].shape[1])
            .type_as(dataset.tensors[1])[indices]
            .clone()
        )
        subset = TensorDataset(dataset.tensors[0][indices].clone(), new_labels)
    else:
        dataset.__dict__[found_key] = labels
        subset = th.utils.data.Subset(dataset, indices)
    return subset
