import json
from copy import deepcopy
import os.path

import numpy as np
import pandas as pd
import torch as th
from lossy.modules import Expression
from lossy.util import set_random_seeds
from kornia.augmentation import RandomCrop
from tensorboardX.writer import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torchvision import transforms

from lossy.condensation.networks import ConvNet
from lossy.datasets import get_dataset
from lossy.scheduler import WarmupBefore
from lossy.losses import soft_cross_entropy_from_logits
from lossy.util import th_to_np
from lossy import data_locations

trange = range
tqdm = lambda x: x


def train_stage(
    trainloader,
    net_features,
    net_clf,
    condensed_and_old_nets,
    optim,
    n_epochs,
    loaders_per_dataset,
    writer,
    lr_schedule,
    crop_pad,
    n_prev_epochs,
    add_distillation_loss,
):
    assert lr_schedule == "cosine"
    if crop_pad > 0:
        aug_fn = RandomCrop(size=(32, 32), padding=crop_pad)
    else:
        assert crop_pad == 0
        aug_fn = nn.Identity()
    for i_epoch in trange(n_epochs):
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
            X = X.cuda()
            y = y.cuda()
            X = aug_fn(X)
            out_features = net_features(X)
            out_clf = net_clf(out_features)
            loss = th.nn.functional.cross_entropy(out_clf, y)
            # impadd
            distill_temperature = 2
            if add_distillation_loss:
                for d in condensed_and_old_nets.values():
                    with th.no_grad():
                        out_old = d["clf"](d["net_features"](X))
                        old_labels = th.softmax(out_old / distill_temperature, dim=1)
                    out_old_with_new_features = d["clf"](out_features)
                    distill_loss = soft_cross_entropy_from_logits(
                        out_old_with_new_features / distill_temperature, old_labels
                    )
                    loss = loss + distill_loss
                loss = loss / (1 + len(condensed_and_old_nets))

            loss.backward()
            # impadd
            for d in condensed_and_old_nets.values():
                condensed_X, condensed_y = d["Xy"]
                out_features = net_features(condensed_X)
                out_clf = d["clf"](out_features)
                condensed_loss = th.nn.functional.cross_entropy(out_clf, condensed_y)
                if add_distillation_loss:
                    for d in condensed_and_old_nets.values():
                        with th.no_grad():
                            out_old = d["clf"](d["net_features"](condensed_X))
                            old_labels = th.softmax(
                                out_old / distill_temperature, dim=1
                            )
                        out_old_with_new_features = d["clf"](out_features)
                        distill_loss = soft_cross_entropy_from_logits(
                            out_old_with_new_features / distill_temperature, old_labels
                        )
                        condensed_loss = condensed_loss + distill_loss
                    condensed_loss = condensed_loss / (1 + len(condensed_and_old_nets))

                condensed_loss.backward()

            optim.step()
            optim.zero_grad(set_to_none=True)
            scheduler.step()

        with th.no_grad():
            net_features.eval()
            print(f"Epoch {i_epoch}")
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
                        if setname in condensed_and_old_nets:
                            this_clf = condensed_and_old_nets[setname]["clf"]
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
                    i_cumulated_epoch = i_epoch + n_prev_epochs
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
    SVHN_exp_folder,
    MNIST_exp_folder,
    n_repetitions_first_stage,
    crop_pad,
    add_distillation_loss,
    debug,
):
    set_random_seeds(np_th_seed, True)
    if same_clf_for_all:
        assert not reset_classifier
        assert train_old_clfs

    momentum = 0.9
    decay = 0.0001
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
            data_locations.pytorch_data,
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

    condensed_per_dataset = {}
    if SVHN_exp_folder is not None:
        condensed_per_dataset["SVHN"] = to_0_1(
            *th.load(os.path.join(SVHN_exp_folder, "res_SVHN_ConvNet_10ipc.pt"))[
                "data"
            ][0]
        )
    if MNIST_exp_folder is not None:
        condensed_per_dataset["MNIST"] = to_0_1(
            *th.load(os.path.join(MNIST_exp_folder, "res_MNIST_ConvNet_10ipc.pt"))[
                "data"
            ][0]
        )

    condensed_results = {}
    if SVHN_exp_folder is not None:
        json_info_path = os.path.join(SVHN_exp_folder, "info.json")
        if os.path.isfile(json_info_path):
            SVHN_config = json.load(open(json_info_path, "r"))
            condensed_results["svhn_bpd"] = SVHN_config["bpd"]
            condensed_results["svhn_acc"] = SVHN_config["acc_mean"]

    if MNIST_exp_folder is not None:
        json_info_path = os.path.join(MNIST_exp_folder, "info.json")
        if os.path.isfile(json_info_path):
            MNIST_config = json.load(open(json_info_path, "r"))
            condensed_results["mnist_bpd"] = MNIST_config["bpd"]
            condensed_results["mnist_acc"] = MNIST_config["acc_mean"]

    condensed_and_old_nets = {}
    epoch_dfs_per_stage = []
    n_epochs_so_far = 0
    for i_dataset, dataset in enumerate(loaders_per_dataset):
        print(f"Stage {i_dataset}")
        optim = th.optim.SGD(
            net.parameters(), lr=lrs[i_dataset], weight_decay=decay, momentum=momentum
        )
        if train_old_clfs and (not same_clf_for_all):
            for d in condensed_and_old_nets.values():
                optim.add_param_group(dict(params=d["clf"].parameters()))
        n_repetitions = 1
        if i_dataset == 0:
            n_repetitions = n_repetitions_first_stage
        for _ in range(n_repetitions):
            # Reset LR, necessary in case of multiple repetitions
            for g in optim.param_groups:
                g["lr"] = lrs[i_dataset]
            epoch_dfs_per_set = train_stage(
                loaders_per_dataset[dataset].train,
                net_features,
                net.classifier,
                condensed_and_old_nets,
                optim,
                n_epochs=n_epochs_per_stage,
                loaders_per_dataset=loaders_per_dataset,
                writer=writer,
                lr_schedule=lr_schedule,
                crop_pad=crop_pad,
                n_prev_epochs=n_epochs_so_far,
                add_distillation_loss=add_distillation_loss,
            )
            n_epochs_so_far += n_epochs_per_stage
        epoch_dfs_per_stage.append(epoch_dfs_per_set)

        if i_dataset < (len(loaders_per_dataset) - 1):
            if same_clf_for_all:
                clf = net.classifier
            else:
                clf = deepcopy(net.classifier)
            if dataset in condensed_per_dataset:
                condensed_and_old_nets[dataset] = dict(
                    Xy=condensed_per_dataset[dataset],
                    net_features=deepcopy(net_features),
                    clf=clf,
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
    writer.flush()
    return results


if __name__ == "__main__":
    n_epochs_per_stage = 50
    lrs = [0.1, 0.01, 0.01]
    np_th_seed = 0
    first_n = None
    train_old_clfs = False
    reset_classifier = True
    same_clf_for_all = False
    lr_schedule = 'cosine'
    n_repetitions_first_stage = 3
    crop_pad = 3
    add_distillation_loss = False
    debug = False
    SVHN_exp_folder = None # to load condensed SVHN dataset
    MNIST_exp_folder = None # to load condensed MNIST dataset
    output_dir = "."

    run_exp(
        output_dir,
        n_epochs_per_stage,
        lrs,
        np_th_seed,
        first_n,
        train_old_clfs,
        reset_classifier,
        same_clf_for_all,
        lr_schedule,
        SVHN_exp_folder,
        MNIST_exp_folder,
        n_repetitions_first_stage,
        crop_pad,
        add_distillation_loss,
        debug,
    )