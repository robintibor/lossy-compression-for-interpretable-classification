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

trange = range
tqdm = lambda x: x


def train_stage(
    trainloader,
    net_features,
    net_clf,
    condensed_and_clfs,
    optim,
    n_epochs,
    loaders_per_dataset,
    start_lr,
    writer,
    lr_schedule,
):
    assert lr_schedule == "cosine"
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
            out_features = net_features(X)
            out_clf = net_clf(out_features)
            loss = th.nn.functional.cross_entropy(out_clf, y)
            loss.backward()
            for (
                condensed_X,
                condensed_y,
            ), condensed_clf in condensed_and_clfs.values():
                out_features = net_features(condensed_X)
                out_clf = condensed_clf(out_features)
                condensed_loss = th.nn.functional.cross_entropy(out_clf, condensed_y)
                condensed_loss.backward()

            optim.step()
            optim.zero_grad(set_to_none=True)
            scheduler.step()
            res = dict(loss=loss.item())
            if len(condensed_and_clfs) > 0:
                res["condensed_loss"] = condensed_loss.item()
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
                for fold_name in ["test"]:  # 'train_det',
                    epochs_df = pd.DataFrame()
                    for X, y in tqdm(getattr(loaders, fold_name)):
                        X = X.cuda()
                        y = y.cuda()
                        out_features = net_features(X)
                        if setname in condensed_and_clfs:
                            this_clf = condensed_and_clfs[setname][1]
                        else:
                            this_clf = net_clf
                        out_clf = this_clf(out_features)
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
                    i_cumulated_epoch = i_epoch + n_epochs * len(condensed_and_clfs)
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
    train_old_clfs,
    reset_classifier,
    same_clf_for_all,
    lr_schedule,
    debug,
):
    set_random_seeds(np_th_seed, True)
    if same_clf_for_all:
        assert not reset_classifier
        assert train_old_clfs

    momentum = 0.9
    decay = 0.0001
    SVHN_exp_id = 267
    MNIST_exp_id = 277
    # USPS_exp_id = 290
    writer = SummaryWriter(output_dir)
    writer.flush()

    dataset_order = ["SVHN", "MNIST", "USPS"]

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
        Expression(lambda x: x - 0.5), net.features, nn.Flatten(start_dim=1)
    )

    to_0_1 = lambda X, y: (th.sigmoid(X).cuda(), y.cuda())

    condensed_per_dataset = {
        "SVHN": to_0_1(
            *th.load(
                f"/home/schirrmr/data/exps/dataset-condensation/mnist-svhn/{SVHN_exp_id:d}/res_SVHN_ConvNet_10ipc.pt"
            )["data"][0]
        ),
        "MNIST": to_0_1(
            *th.load(
                f"/home/schirrmr/data/exps/dataset-condensation/mnist-svhn/{MNIST_exp_id:d}/res_MNIST_ConvNet_10ipc.pt"
            )["data"][0]
        ),
        # 'USPS': to_0_1(*th.load(
        # f'/home/schirrmr/data/exps/dataset-condensation/mnist-svhn/{USPS_exp_id:d}/res_USPS_ConvNet_10ipc.pt')['data'][0]),
    }

    condensed_results = {}
    condensed_results["svhn_bpd"] = json.load(
        open(
            f"/home/schirrmr/data/exps/dataset-condensation/mnist-svhn/{SVHN_exp_id:d}/info.json",
            "r",
        )
    )["bpd"]
    condensed_results["svhn_acc"] = json.load(
        open(
            f"/home/schirrmr/data/exps/dataset-condensation/mnist-svhn/{SVHN_exp_id:d}/info.json",
            "r",
        )
    )["acc_mean"]
    condensed_results["mnist_bpd"] = json.load(
        open(
            f"/home/schirrmr/data/exps/dataset-condensation/mnist-svhn/{MNIST_exp_id:d}/info.json",
            "r",
        )
    )["bpd"]
    condensed_results["mnist_acc"] = json.load(
        open(
            f"/home/schirrmr/data/exps/dataset-condensation/mnist-svhn/{MNIST_exp_id:d}/info.json",
            "r",
        )
    )["acc_mean"]

    condensed_and_clfs = {}
    epoch_dfs_per_stage = []
    for i_dataset, dataset in enumerate(loaders_per_dataset):
        print(f"Stage {i_dataset}")
        # lr will be set inside train_stage during warmup
        optim = th.optim.SGD(
            net.parameters(), lr=lrs[i_dataset], weight_decay=decay, momentum=momentum
        )
        if train_old_clfs and (not same_clf_for_all):
            for _, condensed_clf in condensed_and_clfs.values():
                optim.add_param_group(dict(params=condensed_clf.parameters()))
        epoch_dfs_per_set = train_stage(
            loaders_per_dataset[dataset].train,
            net_features,
            net.classifier,
            condensed_and_clfs,
            optim,
            n_epochs=n_epochs_per_stage,
            loaders_per_dataset=loaders_per_dataset,
            start_lr=lrs[i_dataset],
            writer=writer,
            lr_schedule=lr_schedule,
        )
        epoch_dfs_per_stage.append(epoch_dfs_per_set)

        if i_dataset < (len(loaders_per_dataset) - 1):
            if same_clf_for_all:
                clf = net.classifier
            else:
                clf = deepcopy(net.classifier)
            condensed_and_clfs[dataset] = (
                condensed_per_dataset[dataset],
                clf,
            )
            # maybe optional if to reset, else just deepcopy?
            if reset_classifier:
                net.classifier = nn.Linear(
                    in_features=2048, out_features=10, bias=True
                ).cuda()

    stage_results = {}
    for i_dataset, dataset in enumerate(loaders_per_dataset):
        epoch_dfs_per_set = epoch_dfs_per_stage[i_dataset]
        relevant_epoch_dfs = []
        for setname in dataset_order[: i_dataset + 1]:
            relevant_epoch_dfs.append(epoch_dfs_per_set[setname])
        assert len(relevant_epoch_dfs) == (i_dataset + 1)
        accs = {"train_det": [], "test": []}
        losses = {"train_det": [], "test": []}
        for fold_dfs in relevant_epoch_dfs:
            for fold, this_df in fold_dfs.items():
                acc = (this_df.pred_labels == this_df.y).mean()
                loss = (this_df.loss).mean()
                accs[fold].append(acc)
                losses[fold].append(loss)
        for fold in accs:
            stage_results[f"T{i_dataset + 1}_{fold.replace('_det', '')}_acc"] = np.mean(
                accs[fold]
            )
            stage_results[
                f"T{i_dataset + 1}_{fold.replace('_det', '')}_loss"
            ] = np.mean(losses[fold])
    results = {**condensed_results, **stage_results}
    return results
