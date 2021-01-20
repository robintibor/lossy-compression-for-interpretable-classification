import os

import torch as th
from torch import nn

from lossy.affine import AffineOnChans

th.backends.cudnn.benchmark = True
import numpy as np
from tqdm.autonotebook import tqdm, trange
from rtsutils.util import th_to_np, np_to_th
from braindecode.util import set_random_seeds
from torchvision import transforms
import torchvision

from invertible.view_as import Flatten2d
from invertible.pure_model import NoLogDet
from rtsutils.optim import grads_all_finite
import pandas as pd
from rtsutils.scheduler import CosineAnnealing
from rtsutils.scheduler import ScheduledOptimizer
import higher
import kornia
from itertools import islice
from invertibleeeg.images.unet import UnetGenerator, WrapResidualIdentityUnet
from invertibleeeg.images.moving_batch_norm import MovingBatchNorm2d
from invertible.expression import Expression
from invertibleeeg.images.optim import RAdam
from invertibleeeg.images.datautil import restrict_dataset_to_classes
# train one step with optimized inputs
from tensorboardX import SummaryWriter
from rtsutils.plot import rescale, stack_images_in_rows, create_rgb_image
from torch.optim import Adam
import logging
import matplotlib
#https://stackoverflow.com/a/4935945/1469195
matplotlib.use('Agg')
import seaborn
import matplotlib.pyplot as plt
import sys

log = logging.getLogger(__name__)



def load_data(setname, n_images, train_batch_size=32, test_batch_size=100, i_classes=None):

    if setname == 'CIFAR10':
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR10(
            root='/home/schirrmr/data/pytorch-datasets/data/CIFAR10/',
            train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='/home/schirrmr/data/pytorch-datasets/data/CIFAR10/',
            train=False, download=False, transform=transform)
    elif setname == 'MNIST':
        transform = transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(
            '/home/schirrmr/data/pytorch-datasets/', train=True, download=False,
            transform=transform)
        testset = torchvision.datasets.MNIST(
            '/home/schirrmr/data/pytorch-datasets/', train=False, download=False,
            transform=transform)
    else:
        raise ValueError(f"Unknown setname {setname}")
    if i_classes is not None:
        trainset = restrict_dataset_to_classes(trainset, i_classes, remap_labels=True)
        testset = restrict_dataset_to_classes(testset, i_classes, remap_labels=True)
    if n_images is not None:
        trainset = th.utils.data.Subset(trainset, np.arange(n_images))
        testset = th.utils.data.Subset(testset, np.arange(n_images))
    trainloader = th.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2,
        drop_last=True)
    testloader = th.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2,
        drop_last=False)
    return trainloader, testloader


def preprocess_for_glow(x, n_bits=8):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78
    x = x * 255  # undo ToTensor scaling to [0,1]
    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = th.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    return x


def preprocess_for_clf(x):
    mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
    }

    std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
    }
    th_mean = np_to_th(mean['cifar10'], device=x.device, dtype=np.float32).unsqueeze(
        0).unsqueeze(-1).unsqueeze(-1)
    th_std = np_to_th(std['cifar10'], device=x.device, dtype=np.float32).unsqueeze(
        0).unsqueeze(-1).unsqueeze(-1)
    # if it was greyscale before, keep greyscale
    was_grey_scale = x.shape[1] == 1
    x = (x - th_mean) / th_std
    if was_grey_scale:
        x = th.mean(x, dim=1, keepdim=True)
    return x


def from_glow_to_clf(x):
    # back to original -> [0,1]
    x = x + 0.5
    x = x * 256 / 255
    # now to clf input statistics
    x = preprocess_for_clf(x)
    return x


def load_clf():
    os.sys.path.insert(0, '/home/schirrmr/code/cifar10-clf/wide_resnet/')
    checkpoint = th.load('/home/schirrmr/code/cifar10-clf/wide_resnet/checkpoint/cifar10/wide-resnet-28x10.t7')
    net = checkpoint['net']
    net.eval()
    clf_encoder = nn.Sequential(
        net.conv1,
        net.layer1,
        net.layer2,
        net.layer3,
        net.bn1,
        nn.ReLU(),
        nn.AvgPool2d(8),
        NoLogDet(Flatten2d())
    )
    clf_head = net.linear
    clf = nn.Sequential(clf_encoder, clf_head)
    # just test
    dummy_in = th.randn(2,3,32,32, device='cuda')
    with th.no_grad():
        orig_out = net(dummy_in)
        new_out = clf(dummy_in)
    assert th.allclose(orig_out, new_out)
    return clf


def load_glow(file_name='/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th'):
    model = th.load(file_name)
    from invglow.invertible.init import init_all_modules
    init_all_modules(model, None)
    return model


def is_finite_and_below(x, max_val):
    return th.isfinite(x) & (x < max_val)


def get_X_for_glow_and_for_clf(X):
    X_glow = preprocess_for_glow(X)
    X_glow_noised = X_glow + th.rand_like(X_glow) * 1/256.0
    # rethink this... if not instead just produce x_clf from x...
    X_clf = from_glow_to_clf(X_glow_noised)
    return X_glow_noised, X_clf


def to_plus_minus_one(x):
    return (x * 2) - 1


def get_preproc():
    preproc = WrapResidualIdentityUnet(
        nn.Sequential(
            Expression(to_plus_minus_one),
            UnetGenerator(3,3,num_downs=5, final_nonlin=nn.Identity, norm_layer=AffineOnChans)),
        final_nonlin=nn.Sigmoid(),).cuda()
    return preproc


def train_inner_clf(clf, inner_optim_clf, X_opt_clf_1, X_opt_clf_2, X_orig_clf, y, n_steps,
                    copy_initial_weights):
    with higher.innerloop_ctx(clf, inner_optim_clf, copy_initial_weights=copy_initial_weights) as (
            func_clf, diffopt):

        for i_step in range(n_steps):
            func_clf_out_opt_1 = func_clf(X_opt_clf_1)
            cross_ent_opt_before = th.nn.functional.cross_entropy(func_clf_out_opt_1, y)
            if i_step == 0:
                cross_ent_opt_at_start = cross_ent_opt_before.detach()
            diffopt.step(cross_ent_opt_before)
        return evaluate_clf(func_clf, X_opt_clf_2, X_orig_clf, y) + (cross_ent_opt_at_start,)


def evaluate_clf(clf, X_opt_clf, X_orig_clf, y):
        clf_out_orig = clf(X_orig_clf)
        cross_ent_orig = th.nn.functional.cross_entropy(clf_out_orig, y)
        clf_out_opt = clf(X_opt_clf)
        cross_ent_opt = th.nn.functional.cross_entropy(clf_out_opt, y)

        kldiv_orig_to_inv = th.mean(
            th.sum(
                th.nn.functional.softmax(clf_out_opt, dim=1) *
                (th.nn.functional.log_softmax(clf_out_opt, dim=1) -
                 th.nn.functional.log_softmax(clf_out_orig, dim=1)), dim=1))
        kldiv_inv_to_orig = th.mean(
            th.sum(
                th.nn.functional.softmax(clf_out_orig, dim=1) *
                (th.nn.functional.log_softmax(clf_out_orig, dim=1) -
                 th.nn.functional.log_softmax(clf_out_opt, dim=1)), dim=1))
        return kldiv_orig_to_inv, kldiv_inv_to_orig, cross_ent_orig, cross_ent_opt

def run_exp(
        lr_preproc,
        lr_clf,
        optim_preproc_name,
        class_loss_factor,
        cosine_period,
        np_th_seed,
        debug,
        n_epochs,
        train_classifier,
        train_classifier_together,
        batch_size,
        output_dir):
    hparams = {k:v for k,v in locals().items() if v is not None}

    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    optimize_fixed_inputs = False
    one_step_ahead_loss = False
    train_on_single_batch = False
    assert optim_preproc_name in ['radam', 'adam']
    n_images = None

    if debug:
        n_images = 512


    set_random_seeds(np_th_seed, True)

    log.info("Load data...")
    trainloader, testloader = load_data(n_images, train_batch_size=batch_size)
    log.info("Load classifier...")
    clf = load_clf()
    log.info("Load Glow...")
    gen = load_glow()

    def get_augmentation():
        angle = (th.rand(1) * 10 - 5).cuda()
        contrast_factor = th.exp((th.rand(1) - 0.5) * 0.).cuda()
        sigma = (th.rand(1) * 1. + 1e-5).cuda()

        aug_m = nn.Sequential(
            kornia.Rotate(angle),
            kornia.enhance.AdjustContrast(contrast_factor),
            kornia.filters.GaussianBlur2d(kernel_size=(3, 3),
                                          sigma=(sigma, sigma)))
        aug_m = nn.Identity()

        return aug_m

    def get_inputs(X, preproc):
        X_opt = preproc(X)
        aug_m = get_augmentation()
        X_orig_aug = aug_m(X)
        aug_m = get_augmentation()
        X_opt_aug = aug_m(X_opt)
        _, X_orig_clf = get_X_for_glow_and_for_clf(X_orig_aug)
        _, X_opt_clf_1 = get_X_for_glow_and_for_clf(X_opt_aug)
        aug_m = get_augmentation()
        X_opt_aug_2 = aug_m(X_opt)
        _, X_opt_clf_2 = get_X_for_glow_and_for_clf(X_opt_aug_2)
        return X_opt, X_opt_clf_1, X_opt_clf_2, X_orig_clf

    if optimize_fixed_inputs or train_on_single_batch:
        train_X, train_y = next(trainloader.__iter__())
        # overwrite original test x test y
        test_X = train_X[:32]
        test_y = train_y[:32]
        class InputLoader():
            def __iter__(self):
                for i in range(100):
                    yield train_X[:24], train_y[:24]

            def __len__(self):
                return 100
        trainloader = InputLoader()  # ((X,y) for X,y in ((test_X, test_y),)
    else:
        Xs_ys = list(islice(testloader, 8))
        test_X = th.cat([x for x,y in Xs_ys])[147:147+24]
        test_y = th.cat([y for x,y in Xs_ys])[147:147+24]

    if optimize_fixed_inputs:
        X_to_opt = train_X[:32].cuda().clone().detach().requires_grad_(True)
        X_to_opt.data = th.randn_like(X_to_opt) * 0.01 + 0.5
        preproc = lambda x: X_to_opt
        preproc_params = [X_to_opt]
    else:
        preproc = get_preproc()
        preproc_params = preproc.parameters()
    opt_preproc_class = {'radam': RAdam, 'adam': Adam}[optim_preproc_name]
    opt_preproc = opt_preproc_class([dict(params=preproc_params, lr=lr_preproc,
                                          weight_decay=5e-5, beta=(0.5, 0.99))])
    if cosine_period is not None:
        scheduler = CosineAnnealing(opt_preproc, [cosine_period], schedule_weight_decay=False)
        opt_preproc = ScheduledOptimizer(scheduler, opt_preproc)
    opt_clf = Adam([dict(params=clf.parameters(), lr=lr_clf,
        weight_decay=5e-5, beta=(0., 0.99))])

    for i_epoch in trange(n_epochs + 1):
        if i_epoch > 0:
            preproc.train()
            for i_batch, (X, y) in enumerate(tqdm(trainloader)):
                X = X.cuda()
                y = y.cuda()
                X_opt, X_opt_clf_1, X_opt_clf_2, X_orig_clf = get_inputs(
                    X, preproc, )
                X_opt_for_glow = preprocess_for_glow(X_opt) + th.rand_like(X_opt) * (1 / 256.0)
                _, lp = gen(X_opt_for_glow)
                bpd_opt = -(lp - 3072 * np.log(256)) / (3072 * np.log(2))
                bpd_loss = th.mean(bpd_opt)
                if one_step_ahead_loss:
                    (kldiv_orig_to_inv, kldiv_inv_to_orig,
                     cross_ent_orig, cross_ent_opt, cross_ent_opt_before) = train_inner_clf(
                        clf, opt_clf, X_opt_clf_1, X_opt_clf_2, X_orig_clf, y)
                else:
                    del X_opt_clf_2
                    kldiv_orig_to_inv, kldiv_inv_to_orig, cross_ent_orig, cross_ent_opt = evaluate_clf(
                        clf, X_opt_clf_1, X_orig_clf, y)
                    cross_ent_opt_before = cross_ent_opt

                class_loss = cross_ent_orig + cross_ent_opt + kldiv_inv_to_orig + kldiv_orig_to_inv
                total_weight = 2
                loss = total_weight * (class_loss * class_loss_factor +
                                       bpd_loss * (1 - class_loss_factor))

                opt_preproc.zero_grad()
                loss.backward()
                if grads_all_finite(opt_preproc):
                    opt_preproc.step()
                    if train_classifier_together:
                        opt_clf.step()
                opt_preproc.zero_grad()
                if train_classifier:
                    clf_out = clf(X_opt_clf_1.detach())
                    cross_ent = th.nn.functional.cross_entropy(clf_out, y)
                    opt_clf.zero_grad()
                    cross_ent.backward()
                    opt_clf.step()
                    opt_clf.zero_grad()
        with th.no_grad():
            preproc.eval()
            test_X_opt = preproc(test_X.cuda())
            im = rescale(create_rgb_image(stack_images_in_rows(
                th_to_np(test_X),
                th_to_np(test_X_opt), n_cols=12)), 3)
            writer.add_image('real_and_optimized', np.array(im, dtype=np.uint8).transpose(2,0,1),
                             i_epoch)

            fig = plt.figure(figsize=(6, 3))
            seaborn.distplot(th_to_np(test_X_opt[:, 0].reshape(-1)),
                             color=seaborn.color_palette()[3])
            seaborn.distplot(th_to_np(test_X_opt[:, 1].reshape(-1)),
                             color=seaborn.color_palette()[2])
            seaborn.distplot(th_to_np(test_X_opt[:, 2].reshape(-1)),
                             color=seaborn.color_palette()[0])
            plt.xlabel("Preproc Output Values")
            plt.ylabel("Density")
            writer.add_figure('color_dist', fig, i_epoch)
            plt.close(fig)
            del test_X_opt
            results = {}
            for name, loader in (("train", trainloader), ("test", testloader)):
                df = pd.DataFrame()
                for val_X, val_y in tqdm(loader):
                    val_X = val_X.cuda()
                    val_y = val_y.cuda()
                    X_opt = preproc(val_X)


                    X_orig_for_glow, X_orig_clf = get_X_for_glow_and_for_clf(val_X)
                    _, lp = gen(X_orig_for_glow)
                    bpd_orig = -(lp - 3072 * np.log(256)) / (3072 * np.log(2))
                    X_opt_for_glow, X_opt_clf = get_X_for_glow_and_for_clf(X_opt)
                    _, lp = gen(X_opt_for_glow)
                    bpd_opt = -(lp - 3072 * np.log(256)) / (3072 * np.log(2))
                    out_opt = clf(X_opt_clf)
                    out_orig = clf(X_orig_clf)
                    cross_ent_opt = th.nn.functional.cross_entropy(out_opt, val_y, reduction='none')
                    cross_ent_orig = th.nn.functional.cross_entropy(out_orig, val_y,
                                                                    reduction='none')
                    correct_opt = th_to_np(out_opt.argmax(dim=1) == val_y)
                    correct_orig = th_to_np(out_orig.argmax(dim=1) == val_y)

                    # TODO: make kldiv function
                    # also make evalute_clf function, also return corect opt etc.
                    # makes then just call it here
                    kldiv_orig_to_inv = th.sum(
                            th.nn.functional.softmax(out_opt, dim=1) *
                            (th.nn.functional.log_softmax(out_opt, dim=1) -
                             th.nn.functional.log_softmax(out_orig, dim=1)), dim=1)
                    kldiv_inv_to_orig = th.sum(
                            th.nn.functional.softmax(out_orig, dim=1) *
                            (th.nn.functional.log_softmax(out_orig, dim=1) -
                             th.nn.functional.log_softmax(out_opt, dim=1)), dim=1)
                    class_loss = cross_ent_orig + cross_ent_opt + kldiv_inv_to_orig + kldiv_orig_to_inv
                    total_weight = 2
                    loss = total_weight * (class_loss * class_loss_factor +
                                           bpd_opt * (1 - class_loss_factor))

                    df = pd.concat((df, pd.DataFrame(data=dict(
                        cross_ent_opt=th_to_np(cross_ent_opt),
                        cross_ent_orig=th_to_np(cross_ent_orig),
                        correct_opt=correct_opt,
                        correct_orig=correct_orig,
                        bpd_opt=th_to_np(bpd_opt),
                        bpd_orig=th_to_np(bpd_orig),
                        bpd_diff=th_to_np(bpd_opt) - th_to_np(bpd_orig),
                        class_loss=th_to_np(class_loss),
                        kldiv_inv_to_orig=th_to_np(kldiv_inv_to_orig),
                        kldiv_orig_to_inv=th_to_np(kldiv_orig_to_inv),
                        loss=th_to_np(loss),
                    ))))
                sys.stdout.flush()
                for key, val in df.mean().items():
                    name_key = f"{name}_{key}"
                    print(f"{name_key}: {val:.2f}")
                    writer.add_scalar(name_key, val, i_epoch)
                    results[name_key] = val
                writer.flush()
                sys.stdout.flush()
            if not debug:
                dict_path = os.path.join(output_dir, "preproc_dict.th")
                th.save(preproc.state_dict(), open(dict_path, 'wb'))
                model_path = os.path.join(output_dir, "preproc.th")
                th.save(preproc, open(model_path, 'wb'))

    return results




