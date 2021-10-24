import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from .networks import (
    MLP,
    ConvNet,
    LeNet,
    AlexNet,
    VGG11BN,
    VGG11,
    ResNet18,
    ResNet18BN_AP,
)
from ..augment import TrivialAugmentPerImage


def get_dataset(
    dataset,
    data_path,
    standardize=True,
    split_test_off_train=False,
    mimic_cxr_clip=1.0,  # Clip MIMIC CXR Brightness?
    mimic_cxr_target=None,
):
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
        dst_train = datasets.MNIST(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.MNIST(
            data_path, train=False, download=True, transform=transform
        )
        class_names = [str(c) for c in range(num_classes)]
    elif dataset == "FashionMNIST":
        channel = 3
        im_size = (32, 32)
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
        dst_train = datasets.FashionMNIST(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.FashionMNIST(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes
    elif dataset == "USPS":
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        assert not standardize
        mean = [0.0, 0.0, 0.0]
        std = [1, 1, 1]
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    32,
                ),  # transforms.InterpolationMode.BILINEAR),
                transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
            ]
        )
        dst_train = datasets.USPS(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.USPS(
            data_path, train=False, download=True, transform=transform
        )
        class_names = [str(c) for c in range(num_classes)]
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
        dst_train = datasets.SVHN(
            data_path, split="train", download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.SVHN(
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
        dst_train = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.CIFAR10(
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
        dst_train = datasets.CIFAR100(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes
    elif dataset == "MIMIC-CXR":
        from lossy.datasets import get_dataset as get_dataset_lossy
        (
            channel,
            im_size,
            num_classes,
            class_names,
            trainloader,
            train_det_loader,
            testloader,
        ) =  get_dataset_lossy(dataset,
            data_path,
            batch_size=64,
            standardize=standardize,
            split_test_off_train=split_test_off_train,
            first_n=None,
            eval_batch_size=256,
            i_classes=None,
            mimic_cxr_clip=mimic_cxr_clip,
            mimic_cxr_target=mimic_cxr_target)

        return (
            channel,
            im_size,
            num_classes,
            class_names,
            None,
            None,
            trainloader.dataset,
            testloader.dataset,
            testloader,
        )

    else:
        exit("unknown dataset: %s" % dataset)

    if split_test_off_train:
        n_train = len(dst_train)
        n_split = int(np.ceil(n_train * 0.8))
        dst_test = torch.utils.data.Subset(dst_train, np.arange(n_split, n_train))
        dst_train = torch.utils.data.Subset(dst_train, np.arange(0, n_split))

    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=256, shuffle=False, num_workers=2
    )
    return (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
    )


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = (
        128,
        3,
        "relu",
        "instancenorm",
        "avgpooling",
    )
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(
    model,
    channel,
    num_classes,
    im_size=(32, 32),
    net_norm_override=None,
    net_act_override=None,
):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    if net_norm_override is not None:
        net_norm = net_norm_override
    if net_act_override is not None:
        net_act = net_act_override

    if model == "MLP":
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == "ConvNet":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "LeNet":
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == "AlexNet":
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == "VGG11":
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == "VGG11BN":
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == "ResNet18":
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == "ResNet18BN_AP":
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)

    elif model == "ConvNetD1":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=1,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetD2":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=2,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetD3":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=3,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetD4":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )

    elif model == "ConvNetW32":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=32,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetW64":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=64,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetW128":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=128,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetW256":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=256,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
        )

    elif model == "ConvNetAS":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act="sigmoid",
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetAR":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act="relu",
            net_norm=net_norm,
            net_pooling=net_pooling,
        )
    elif model == "ConvNetAL":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act="leakyrelu",
            net_norm=net_norm,
            net_pooling=net_pooling,
        )

    elif model == "ConvNetNN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="none",
            net_pooling=net_pooling,
        )
    elif model == "ConvNetBN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="batchnorm",
            net_pooling=net_pooling,
        )
    elif model == "ConvNetLN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="layernorm",
            net_pooling=net_pooling,
        )
    elif model == "ConvNetIN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="instancenorm",
            net_pooling=net_pooling,
        )
    elif model == "ConvNetGN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="groupnorm",
            net_pooling=net_pooling,
        )

    elif model == "ConvNetNP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling="none",
        )
    elif model == "ConvNetMP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling="maxpooling",
        )
    elif model == "ConvNetAP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling="avgpooling",
        )

    else:
        net = None
        exit("DC error: unknown model")

    gpu_num = torch.cuda.device_count()
    if gpu_num > 0:
        device = "cuda"
        if gpu_num > 1:
            net = nn.DataParallel(net)
    else:
        device = "cpu"
    net = net.to(device)

    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = "do nothing"
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(
        1
        - torch.sum(gwr * gws, dim=-1)
        / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
    )
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, dis_metric, device):
    dis = torch.tensor(0.0).to(device)

    if dis_metric == "ours":
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        exit("DC error: unknown distance function")

    return dis


def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit("DC error: loop hyper-parameters are not defined for %d ipc" % ipc)
    return outer_loop, inner_loop


def epoch(
    mode,
    dataloader,
    net,
    optimizer,
    criterion,
    param_augment,
    device,
    image_converter,
    trivial_augment=None,
    same_aug_across_batch=None,
):
    assert trivial_augment is not None
    assert same_aug_across_batch is not None
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    if mode == "train":
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img_orig = datum[0].float().to(device)
        if (mode == "train") and trivial_augment:
            aug_m = TrivialAugmentPerImage(
                img_orig.shape[0],
                num_magnitude_bins=31,
                std_aug_magnitude=None,
                extra_augs=True,
                same_across_batch=same_aug_across_batch,
            )
            img_orig = aug_m(img_orig)
        img = image_converter.img_orig_to_clf(img_orig)
        if mode == "train" and (param_augment != None) and (not trivial_augment):
            img = augment(img, param_augment, device=device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(
            np.equal(
                np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()
            )
        )

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == "train":
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(
    it_eval,
    net,
    images_train,
    labels_train,
    testloader,
    learningrate,
    batchsize_train,
    param_augment,
    device,
    Epoch=600,
    image_converter=None,
    trivial_augment=None,
    same_aug_across_batch=None,
):
    assert trivial_augment is not None
    assert same_aug_across_batch is not None
    net = net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)
    lr = float(learningrate)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )
    criterion = nn.CrossEntropyLoss().to(device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=batchsize_train, shuffle=True, num_workers=0
    )

    start = time.time()
    for ep in range(Epoch + 1):
        loss_train, acc_train = epoch(
            "train",
            trainloader,
            net,
            optimizer,
            criterion,
            param_augment,
            device,
            image_converter=image_converter,
            trivial_augment=trivial_augment,
            same_aug_across_batch=same_aug_across_batch,
        )
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(
                net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
            )

    time_train = time.time() - start
    loss_test, acc_test = epoch(
        "test",
        testloader,
        net,
        optimizer,
        criterion,
        param_augment,
        device,
        image_converter=image_converter,
        trivial_augment=trivial_augment,
        same_aug_across_batch=same_aug_across_batch,
    )
    print(
        "%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f"
        % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test)
    )

    return net, acc_train, acc_test


def augment(images, param_augment, device):
    # This can be sped up in the future.

    if param_augment != None and param_augment["strategy"] != "none":
        scale = param_augment["scale"]
        crop = param_augment["crop"]
        rotate = param_augment["rotate"]
        noise = param_augment["noise"]
        strategy = param_augment["strategy"]

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:, c])))

        def cropfun(i):
            im_ = torch.zeros(
                shape[1],
                shape[2] + crop * 2,
                shape[3] + crop * 2,
                dtype=torch.float,
                device=device,
            )
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop : crop + shape[2], crop : crop + shape[3]] = images[i]
            r, c = (
                np.random.permutation(crop * 2)[0],
                np.random.permutation(crop * 2)[0],
            )
            images[i] = im_[:, r : r + shape[2], c : c + shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(
                images[i : i + 1],
                [h, w],
            )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r : r + h, c : c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r : r + shape[2], c : c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(
                images[i].cpu().data.numpy(),
                angle=np.random.randint(-rotate, rotate),
                axes=(-2, -1),
                cval=np.mean(mean),
            )
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(
                im_[:, r : r + shape[-2], c : c + shape[-1]],
                dtype=torch.float,
                device=device,
            )

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(
                shape[1:], dtype=torch.float, device=device
            )

        augs = strategy.split("_")

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[
                0
            ]  # randomly implement one augmentation
            if choice == "crop":
                cropfun(i)
            elif choice == "scale":
                scalefun(i)
            elif choice == "rotate":
                rotatefun(i)
            elif choice == "noise":
                noisefun(i)

    return images


def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    param_augment = dict()
    param_augment["crop"] = 4
    param_augment["scale"] = 0.2
    param_augment["rotate"] = 45
    param_augment["noise"] = 0.001
    param_augment["strategy"] = "none"

    if dataset == "MNIST":
        param_augment["strategy"] = "crop_scale_rotate"

    if model_eval in [
        "ConvNetBN"
    ]:  # Data augmentation makes model training with Batch Norm layer easier.
        param_augment["strategy"] = "crop_noise"

    return param_augment


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == "M":  # multiple architectures
        model_eval_pool = ["MLP", "ConvNet", "LeNet", "AlexNet", "VGG11", "ResNet18"]
    elif eval_mode == "W":  # ablation study on network width
        model_eval_pool = ["ConvNetW32", "ConvNetW64", "ConvNetW128", "ConvNetW256"]
    elif eval_mode == "D":  # ablation study on network depth
        model_eval_pool = ["ConvNetD1", "ConvNetD2", "ConvNetD3", "ConvNetD4"]
    elif eval_mode == "A":  # ablation study on network activation function
        model_eval_pool = ["ConvNetAS", "ConvNetAR", "ConvNetAL"]
    elif eval_mode == "P":  # ablation study on network pooling layer
        model_eval_pool = ["ConvNetNP", "ConvNetMP", "ConvNetAP"]
    elif eval_mode == "N":  # ablation study on network normalization layer
        model_eval_pool = [
            "ConvNetNN",
            "ConvNetBN",
            "ConvNetLN",
            "ConvNetIN",
            "ConvNetGN",
        ]
    elif eval_mode == "S":  # itself
        model_eval_pool = [model[: model.index("BN")]] if "BN" in model else [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool
