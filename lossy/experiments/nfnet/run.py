import logging

import kornia
import numpy as np
import torch
import torch as th
from braindecode.util import set_random_seeds
from nfnets.base import ScaledStdConv2d
from nfnets.models.resnet import NFResNetCIFAR, BasicBlock

# train one step with optimized inputs
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from lossy.datasets import get_dataset
from rtsutils.util import np_to_th, th_to_np
from nfnets.models.resnet import activation_fn
from rtsutils.nb_util import NoOpResults
from tqdm import tqdm, trange
import sys  # impadd
from lossy.scheduler import AlsoScheduleWeightDecay, WarmupBefore
import os.path

log = logging.getLogger(__name__)


def run_exp(
    zero_init_residual,
    initialization,
    restart_epochs,
    adjust_betas,
    lr,
    weight_decay,
    n_epochs,
    batch_size,
    split_test_off_train,
    np_th_seed,
    first_n,
    optim_type,
    n_start_filters,
    bias_for_conv,
    n_warmup_epochs,
    drop_path,
    debug,
    output_dir,
):
    assert optim_type in ["adam", "adamw"]
    hparams = {k: v for k, v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    set_random_seeds(np_th_seed, True)
    data_path = "/home/schirrmr/data/pytorch-datasets/data/CIFAR10/"
    dataset = "CIFAR10"
    split_test_off_train = False
    (
        channel,
        im_size,
        num_classes,
        class_names,
        trainloader,
        train_det_loader,
        testloader,
    ) = get_dataset(
        dataset,
        data_path,
        batch_size=batch_size,
        standardize=False,
        split_test_off_train=split_test_off_train,
        first_n=first_n,
    )
    aug_m = nn.Sequential(
        kornia.augmentation.RandomCrop(
            (32, 32),
            padding=4,
        ),
        kornia.augmentation.RandomHorizontalFlip(),
    )
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    normalize = kornia.augmentation.Normalize(
        mean=np_to_th(mean, device="cpu", dtype=np.float32),
        std=np_to_th(std, device="cpu", dtype=np.float32),
    )
    set_random_seeds(3434, True)
    nf_net = NFResNetCIFAR(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=10,
        activation="elu",
        alpha=0.2,
        beta=1.0,
        zero_init_residual=zero_init_residual,
        base_conv=ScaledStdConv2d,
        n_start_filters=n_start_filters,
        bias_for_conv=bias_for_conv,
        drop_path=drop_path,
    )
    nf_net = nf_net.cuda()

    params_with_weight_decay = []
    params_without_weight_decay = []
    for name, param in nf_net.named_parameters():
        if "weight" in name or "gain" in name:
            params_with_weight_decay.append(param)
        else:
            assert "bias" in name
            params_without_weight_decay.append(param)

    if optim_type == "adam":
        opt_nf_net = torch.optim.Adam(
            [
                dict(params=params_with_weight_decay, weight_decay=weight_decay),
                dict(params=params_without_weight_decay, weight_decay=0),
            ],
            lr=lr,
        )
    else:
        assert optim_type == "adamw"
        opt_nf_net = torch.optim.AdamW(
            [
                dict(params=params_with_weight_decay, weight_decay=weight_decay),
                dict(params=params_without_weight_decay, weight_decay=0),
            ],
            lr=lr,
        )

    if restart_epochs is not None:
        scheduler = CosineAnnealingWarmRestarts(
            opt_nf_net, T_0=len(trainloader) * restart_epochs
        )
    else:
        # no scheduling
        scheduler = LambdaLR(opt_nf_net, lr_lambda=lambda *args: 1)
    # impadd
    if n_warmup_epochs > 0:
        scheduler = WarmupBefore(
            scheduler, n_warmup_steps=len(trainloader) * n_warmup_epochs
        )
    # end impadd
    if opt_nf_net.__class__.__name__ == "AdamW":
        scheduler = AlsoScheduleWeightDecay(scheduler)
    else:
        assert optim_type == "adam"

    for name, module in nf_net.named_modules():
        if "Conv2d" in module.__class__.__name__:
            assert hasattr(module, "weight")
            nn.init.__dict__[initialization + "_"](module.weight)
    if hasattr(module, "bias"):
        module.bias.data.zero_()

    from itertools import islice

    if adjust_betas:
        # actually should et the zero init blcoks to 1 before, and afterwards to zero agian... according to paper logic
        for name, module in nf_net.named_modules():
            if module.__class__.__name__ == "ScalarMultiply":
                module.scalar.data[:] = 1
        with th.no_grad():
            init_X = th.cat(
                [normalize(aug_m(X.cuda())) for X, y in islice(trainloader, 10)]
            )

        def adjust_beta(module, input):
            assert len(input) == 1
            out = activation_fn[module.activation](input[0])
            std = out.std(dim=(1, 2, 3)).mean()

            module.beta = 1 / float(std)
            print(std)

        handles = []
        for m in nf_net.modules():
            if m.__class__.__name__ == "BasicBlock":
                handle = m.register_forward_pre_hook(adjust_beta)
                handles.append(handle)

        try:
            with th.no_grad():
                nf_net(init_X)
        finally:
            for handle in handles:
                handle.remove()
        for name, module in nf_net.named_modules():
            if module.__class__.__name__ == "ScalarMultiply":
                module.scalar.data[:] = 0

    nb_res = NoOpResults(0.95)
    for i_epoch in trange(n_epochs + n_warmup_epochs):
        print(f"Epoch {i_epoch:d}")
        nf_net.train()
        for X, y in tqdm(trainloader):
            X = X.cuda()
            y = y.cuda()
            X_aug = normalize(aug_m(X))
            out = nf_net(X_aug.cuda())
            loss = th.nn.functional.cross_entropy(out, y)
            opt_nf_net.zero_grad(set_to_none=True)
            loss.backward()
            # Maybe keep grads for analysis?
            opt_nf_net.step()
            opt_nf_net.zero_grad(set_to_none=True)
            scheduler.step()
            nb_res.collect(loss=loss.item())
            nb_res.print()
            # with nb_res.output_area('lr'):
            # print(f"LR: {opt_nf_net.param_groups[0]['lr']:.2E}")
        with nb_res.output_area("accs_losses"):
            nf_net.eval()
            results = {}
            with torch.no_grad():
                for set_name, loader in (
                    ("train", train_det_loader),
                    ("test", testloader),
                ):
                    all_preds = []
                    all_ys = []
                    all_losses = []
                    for X, y in tqdm(loader):
                        X = X.cuda()
                        y = y.cuda()
                        X = normalize(X)
                        preds = nf_net(X)
                        all_preds.append(th_to_np(preds))
                        all_ys.append(th_to_np(y))
                        all_losses.append(
                            th_to_np(
                                th.nn.functional.cross_entropy(
                                    preds, y, reduction="none"
                                )
                            )
                        )
                    all_preds = np.concatenate(all_preds)
                    all_ys = np.concatenate(all_ys)
                    all_losses = np.concatenate(all_losses)
                    acc = np.mean(all_preds.argmax(axis=1) == all_ys)
                    mean_loss = np.mean(all_losses)
                    print(f"{set_name.capitalize()} Acc:  {acc:.1%}")
                    print(f"{set_name.capitalize()} Loss: {mean_loss: .2f}")
                    writer.add_scalar(set_name + "_acc", acc * 100, i_epoch)
                    writer.add_scalar(set_name + "_loss", mean_loss, i_epoch)
                    results[set_name + "_acc"] = acc * 100
                    results[set_name + "_loss"] = mean_loss

                writer.add_scalar(
                    "learning_rate", opt_nf_net.param_groups[0]["lr"], i_epoch
                )
                writer.add_scalar(
                    "weight_decay", opt_nf_net.param_groups[0]["weight_decay"], i_epoch
                )
                writer.flush()
                sys.stdout.flush()
    writer.close()
    return results
