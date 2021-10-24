import json
import logging
import os.path
import sys

import kornia
import numpy as np
import pandas as pd
import torch
import torch as th
from lossy.modules import Expression
from lossy.util import set_random_seeds
from tensorboardX.writer import SummaryWriter
from torch import nn

from lossy import data_locations
from lossy.affine import AffineOnChans
from lossy.augment import FixedAugment
from lossy.datasets import get_dataset
from lossy.image2image import WrapResidualIdentityUnet, UnetGenerator
from lossy.image_convert import add_glow_noise_to_0_1
from lossy.image_convert import to_plus_minus_one
from lossy.util import np_to_th, th_to_np

log = logging.getLogger(__name__)


def run_exp(
    output_dir,
    first_n,
    saved_exp_folder,
    n_epochs,
    init_pretrained_clf,
    lr_clf,
    np_th_seed,
    weight_decay,
    debug,
    save_models,
    with_batchnorm,
    noise_on_simplifier,
):

    writer = SummaryWriter(output_dir)
    hparams = {k: v for k, v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    tqdm = lambda x: x
    trange = range

    set_random_seeds(np_th_seed, True)

    config = json.load(open(os.path.join(saved_exp_folder, "config.json"), "r"))
    noise_augment_level = config["noise_augment_level"]
    saved_model_folder = config["saved_model_folder"]
    dataset = config["dataset"]

    assert config["noise_after_simplifier"]
    batch_size = config["batch_size"]
    split_test_off_train = False

    log.info("Load data...")
    data_path = data_locations.pytorch_data
    (
        channel,
        im_size,
        num_classes,
        class_names,
        trainloader,
        train_det_loader,
        testloader,
    ) = get_dataset(
        dataset.upper(),
        data_path,
        batch_size=batch_size,
        standardize=False,
        split_test_off_train=split_test_off_train,
        first_n=first_n,
    )

    depth = config.get("depth", 28)
    widen_factor = config.get("widen_factor", 10)
    dropout = 0.3
    activation = "elu"  # was relu in past
    if saved_model_folder is not None:
        saved_model_config = json.load(
            open(os.path.join(saved_model_folder, "config.json"), "r")
        )
        depth = saved_model_config["depth"]
        widen_factor = saved_model_config["widen_factor"]
        dropout = saved_model_config["dropout"]
        activation = saved_model_config.get("activation", "relu")  # default was relu
        assert saved_model_config.get("dataset", "cifar10") == dataset
    if init_pretrained_clf:
        assert config["save_models"]

    log.info("Create classifier...")
    # Create model

    from lossy import wide_nf_net

    if not with_batchnorm:
        from lossy.wide_nf_net import conv_init, Wide_NFResNet

        model = Wide_NFResNet(
            depth, widen_factor, dropout, num_classes, activation=activation
        ).cuda()
        model.apply(conv_init)
    else:
        activation = "relu"  # overwrite for wide resnet for now
        from lossy.wide_resnet import Wide_ResNet, conv_init

        model = Wide_ResNet(
            depth, widen_factor, dropout, num_classes, activation=activation
        ).cuda()
        model.apply(conv_init)
    log.info("Create classifier...")
    mean = wide_nf_net.mean[dataset]
    std = wide_nf_net.std[dataset]
    normalize = kornia.augmentation.Normalize(
        mean=np_to_th(mean, device="cpu", dtype=np.float32),
        std=np_to_th(std, device="cpu", dtype=np.float32),
    )

    clf = nn.Sequential(normalize, model)
    clf = clf.cuda()
    if init_pretrained_clf:
        saved_clf_state_dict = th.load(
            os.path.join(saved_exp_folder, "clf_state_dict.th")
        )
        clf.load_state_dict(saved_clf_state_dict)

    log.info("Create preprocessor...")
    # Create preproc

    assert config["residual_preproc"]
    preproc = WrapResidualIdentityUnet(
        nn.Sequential(
            Expression(to_plus_minus_one),
            UnetGenerator(
                3,
                3,
                num_downs=5,
                final_nonlin=nn.Identity,
                norm_layer=AffineOnChans,
                nonlin_down=nn.ELU,
                nonlin_up=nn.ELU,
            ),
        ),
        final_nonlin=nn.Sigmoid(),
    ).cuda()
    preproc = nn.Sequential(preproc, Expression(add_glow_noise_to_0_1))
    preproc.load_state_dict(
        th.load(os.path.join(saved_exp_folder, "preproc_state_dict.th"))
    )

    # without noise
    if not noise_on_simplifier:
        preproc = preproc[0]
    preproc.eval()

    log.info("Create optimizers...")
    params_with_weight_decay = []
    params_without_weight_decay = []
    for name, param in clf.named_parameters():
        if "weight" in name or "gain" in name:
            params_with_weight_decay.append(param)
        else:
            assert "bias" in name
            params_without_weight_decay.append(param)

    beta_clf = (0.9, 0.995)
    beta_preproc = (0.5, 0.99)

    opt_clf = torch.optim.AdamW(
        [
            dict(params=params_with_weight_decay, weight_decay=weight_decay),
            dict(params=params_without_weight_decay, weight_decay=0),
        ],
        lr=lr_clf,
        betas=beta_clf,
    )

    def get_aug_m(X_shape, noise_augment_level):
        noise = th.randn(*X_shape, device="cuda") * noise_augment_level
        aug_m = nn.Sequential()
        aug_m.add_module(
            "crop",
            FixedAugment(
                kornia.augmentation.RandomCrop(
                    (32, 32),
                    padding=4,
                ),
                X_shape,
            ),
        )

        if dataset in ["cifar10", "cifar100"]:
            aug_m.add_module(
                "hflip",
                FixedAugment(kornia.augmentation.RandomHorizontalFlip(), X_shape),
            )
        else:
            assert dataset in ["mnist", "fashionmnist", "svhn"]
        aug_m.add_module("noise", Expression(lambda x: x + noise))

        return aug_m

    # Construct fixed train/test sets

    def get_tensor_loaders(loader, preproc, batch_size):
        all_X = []
        all_y = []
        for X, y in tqdm(loader):
            X = X.cuda()
            with th.no_grad():
                simple_X = preproc(X)
            all_X.append(simple_X.cpu())
            all_y.append(y.cpu())
        tensor_set = th.utils.data.TensorDataset(th.cat(all_X), th.cat(all_y))
        shuffled_loader = torch.utils.data.DataLoader(
            tensor_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        det_loader = torch.utils.data.DataLoader(
            tensor_set,
            batch_size=256,
            shuffle=False,
            num_workers=2,
            drop_last=False,
        )
        return shuffled_loader, det_loader

    trainloader_tensors, train_det_loader_tensors = get_tensor_loaders(
        train_det_loader, preproc, batch_size
    )
    _, testloader_tensors = get_tensor_loaders(testloader, preproc, batch_size)
    for i_epoch in trange(n_epochs):
        for X, y in tqdm(trainloader_tensors):
            clf.train()
            X = X.cuda()
            y = y.cuda()
            with th.no_grad():
                aug_m = get_aug_m(X.shape, noise_augment_level)
            # since this is how it was trained before
            X = add_glow_noise_to_0_1(X)
            X_aug = aug_m(X)
            out = clf(X_aug)
            clf_loss = th.nn.functional.cross_entropy(out, y)
            opt_clf.zero_grad(set_to_none=True)
            clf_loss.backward()
            opt_clf.step()
            opt_clf.zero_grad(set_to_none=True)

        clf.eval()
        results = {}
        for loader_name, loader in (
            ("train_preproc", train_det_loader_tensors),
            ("train", train_det_loader),
            ("test_preproc", testloader_tensors),
            ("test", testloader),
        ):
            eval_df = pd.DataFrame()
            for X, y in tqdm(loader):
                X = X.cuda()
                y = y.cuda()
                if "preproc" in loader_name:
                    X = X + (1 / (2 * 255.0))
                out = clf(X)
                loss = th.nn.functional.cross_entropy(out, y, reduction="none")
                eval_df = pd.concat(
                    (
                        eval_df,
                        pd.DataFrame(
                            dict(
                                y=th_to_np(y),
                                pred_label=th_to_np(out.argmax(dim=1)),
                                loss=th_to_np(loss),
                            )
                        ),
                    )
                )
            acc = (eval_df.pred_label == eval_df.y).mean()
            mean_loss = np.mean(eval_df.loss)
            key = loader_name
            print(f"{key.capitalize()} Acc:  {acc:.1%}")
            print(f"{key.capitalize()} Loss: {mean_loss: .2f}")
            writer.add_scalar(key + "_acc", acc * 100, i_epoch)
            writer.add_scalar(key + "_loss", mean_loss, i_epoch)
            results[key + "_acc"] = acc * 100
            results[key + "_loss"] = mean_loss
            writer.flush()
            sys.stdout.flush()
            if save_models and not debug:
                th.save(clf.state_dict(), os.path.join(output_dir, "clf_state_dict.th"))
    return results


if __name__ == "__main__":
    n_epochs = 100
    init_pretrained_clf = False
    np_th_seed = 0
    save_models = True
    with_batchnorm = False
    noise_on_simplifier = True
    weight_decay = 1e-05
    lr_clf = 0.0005
    first_n = None
    saved_exp_folder =  None  # has to be set to model that was saved from one_step/run.py
    debug = False

    output_dir = "."
    run_exp(
        output_dir,
        first_n,
        saved_exp_folder,
        n_epochs,
        init_pretrained_clf,
        lr_clf,
        np_th_seed,
        weight_decay,
        debug,
        save_models,
        with_batchnorm,
        noise_on_simplifier,
    )
