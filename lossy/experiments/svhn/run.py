import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch as th
from braindecode.models.modules import Expression
from braindecode.util import set_random_seeds
from tensorboardX.writer import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torchvision import transforms

from datasetcondensation.networks import ConvNet
from lossy.datasets import get_dataset
from lossy.scheduler import WarmupBefore
from rtsutils.nb_util import NoOpResults
from rtsutils.util import th_to_np
from kornia.augmentation import RandomCrop

trange = range
tqdm = lambda x: x


def train_stage(
    trainloader,
    net_features,
    net_clf,
    optim,
    n_epochs,
    loaders_per_dataset,
    writer,
    lr_schedule,
    n_previous_epochs,
    crop_pad,
):
    aug_fn = RandomCrop(size=(32, 32), padding=crop_pad)
    for i_epoch in trange(n_epochs):
        nb_res = NoOpResults(0.98)
        if lr_schedule == "cosine":
            scheduler = CosineAnnealingWarmRestarts(
                optim, T_0=len(trainloader) * (n_epochs - 1)
            )
        else:
            # No scheduling
            scheduler = LambdaLR(optim, lr_lambda=lambda *args: 1)
        scheduler = WarmupBefore(scheduler, n_warmup_steps=len(trainloader) * 1)
        net_features.train()
        for i_batch, (X, y) in enumerate(tqdm(trainloader)):
            optim.zero_grad(set_to_none=True)
            # if i_epoch == 0:
            #    for g in optim.param_groups:
            #        g['lr'] = start_lr * (i_batch / len(trainloader))
            X = X.cuda()
            y = y.cuda()
            X = aug_fn(X)
            out_features = net_features(X)
            out_clf = net_clf(out_features)
            loss = th.nn.functional.cross_entropy(out_clf, y)
            loss.backward()

            optim.step()
            optim.zero_grad(set_to_none=True)
            scheduler.step()
            res = dict(loss=loss.item())
            nb_res.collect(**res)
            nb_res.print()
        # if i_epoch % 5 == 4:
        #    optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * 0.5

        with th.no_grad():
            net_features.eval()
            print(f"Epoch {i_epoch}")
            nb_res.plot_df()
            epoch_dfs_per_set = {}
            for setname, loaders in loaders_per_dataset.items():
                epochs_df_per_fold = {}
                fold_names = (
                    ["test"] if i_epoch < (n_epochs - 1) else ["train_det", "test"]
                )
                for fold_name in fold_names:
                    epochs_df = pd.DataFrame()
                    for X, y in tqdm(getattr(loaders, fold_name)):
                        X = X.cuda()
                        y = y.cuda()
                        out_features = net_features(X)
                        out_clf = net_clf(out_features)
                        pred_labels = out_clf.argmax(dim=1)
                        losses = th.nn.functional.cross_entropy(
                            out_clf, y, reduction="none"
                        )
                        epochs_df = pd.concat(
                            (
                                epochs_df,
                                pd.DataFrame(
                                    data=dict(
                                        pred_labels=th_to_np(pred_labels),
                                        y=th_to_np(y),
                                        loss=th_to_np(losses),
                                    )
                                ),
                            )
                        )
                    epochs_df_per_fold[fold_name] = epochs_df
                    acc = (epochs_df.pred_labels == epochs_df.y).mean()
                    loss = (epochs_df.loss).mean()
                    print(f"{setname:6s} {fold_name[:5].capitalize()} Acc:  {acc:.1%}")
                    print(f"{setname:6s} {fold_name[:5].capitalize()} Loss: {loss:2f}")
                    i_cumulated_epoch = i_epoch + n_previous_epochs
                    writer.add_scalar(
                        f"{setname.lower()}_{fold_name[:5]}_acc", acc, i_cumulated_epoch
                    )
                    writer.add_scalar(
                        f"{setname.lower()}_{fold_name[:5]}_loss",
                        loss,
                        i_cumulated_epoch,
                    )
                    writer.flush()

                epoch_dfs_per_set[setname] = epochs_df_per_fold
    return epoch_dfs_per_set


def run_exp(
    output_dir,
    n_epochs_per_stage,
    lrs,
    np_th_seed,
    first_n,
    weight_decay,
    assumed_std,
    crop_pad,
    debug,
):
    set_random_seeds(np_th_seed, True)

    momentum = 0.9
    decay = weight_decay
    lr_schedule = "cosine"
    writer = SummaryWriter(output_dir)
    writer.flush()

    dataset_order = [
        "SVHN",
    ]

    from collections import namedtuple

    loaders_per_dataset = {}
    for dataset_name in dataset_order:
        (
            channel,
            im_size,
            num_classes,
            class_names,
            trainloader,
            train_det_loader,
            testloader,
        ) = get_dataset(
            dataset_name,
            "/home/schirrmr/data/pytorch-datasets/",
            batch_size=512,
            standardize=False,
            first_n=first_n,
        )
        if first_n is None:
            trainloader.dataset.transforms.transform.transforms.insert(
                0, transforms.RandomAffine(0, translate=(0.1, 0.1))
            )
        else:
            trainloader.dataset.dataset.transforms.transform.transforms.insert(
                0, transforms.RandomAffine(0, translate=(0.1, 0.1))
            )

        loaders_per_dataset[dataset_name] = namedtuple(
            "Loaders", ["train", "train_det", "test"]
        )(trainloader, train_det_loader, testloader)

    net_width, net_depth, net_act, net_norm, net_pooling = (
        128,
        3,
        "relu",
        "instancenorm",
        "avgpooling",
    )
    net = ConvNet(
        channel,
        num_classes,
        net_width=net_width,
        net_depth=net_depth,
        net_act=net_act,
        net_norm=net_norm,
        net_pooling=net_pooling,
        im_size=im_size,
    ).cuda()
    net_features = nn.Sequential(
        Expression(lambda x: (x - 0.5) / assumed_std),
        net.features,
        nn.Flatten(start_dim=1),
    )

    epoch_dfs_per_stage = []
    # lr will be set inside train_stage during warmup
    optim = th.optim.SGD(
        net.parameters(), lr=lrs[0], weight_decay=decay, momentum=momentum
    )
    for i_stage, lr in enumerate(lrs):
        print(f"Stage {i_stage}")
        for g in optim.param_groups:
            g["lr"] = lr
        epoch_dfs_per_set = train_stage(
            loaders_per_dataset["SVHN"].train,
            net_features,
            net.classifier,
            optim,
            n_epochs=n_epochs_per_stage,
            loaders_per_dataset=loaders_per_dataset,
            writer=writer,
            lr_schedule=lr_schedule,
            n_previous_epochs=n_epochs_per_stage * i_stage,
            crop_pad=crop_pad,
        )
        epoch_dfs_per_stage.append(epoch_dfs_per_set)

    stage_results = {}
    for i_stage in range(len(lrs)):
        epoch_dfs_this_stage = epoch_dfs_per_stage[i_stage]
        relevant_epoch_dfs = [epoch_dfs_this_stage["SVHN"]]
        accs = {"train_det": [], "test": []}
        losses = {"train_det": [], "test": []}
        #accs = {"test": []}
        #losses = {"test": []}
        for fold_dfs in relevant_epoch_dfs:
            for fold, this_df in fold_dfs.items():
                acc = (this_df.pred_labels == this_df.y).mean()
                loss = (this_df.loss).mean()
                accs[fold].append(acc)
                losses[fold].append(loss)
        for fold in accs:
            stage_results[f"T{i_stage + 1}_{fold.replace('_det', '')}_acc"] = np.mean(
                accs[fold]
            )
            stage_results[f"T{i_stage + 1}_{fold.replace('_det', '')}_loss"] = np.mean(
                losses[fold]
            )
        if i_stage == len(lrs) - 1:
            stage_results[f"final_{fold.replace('_det', '')}_acc"] = np.mean(
                accs[fold]
            )
            stage_results[f"final_{fold.replace('_det', '')}_loss"] = np.mean(
                losses[fold]
            )

    results = {**stage_results}
    return results
