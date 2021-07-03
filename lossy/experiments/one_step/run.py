import os.path
import sys
from itertools import islice

import kornia
import numpy as np
import torch
import torch as th
from braindecode.models.modules import Expression
from braindecode.util import set_random_seeds
from tensorboardX.writer import SummaryWriter
from torch import nn
from torchvision.utils import save_image

import higher
from lossy.affine import AffineOnChans
from lossy.augment import FixedAugment
from lossy.datasets import get_dataset
from lossy.image2image import WrapResidualIdentityUnet, UnetGenerator
from lossy.image_convert import img_0_1_to_glow_img
from lossy.util import weighted_sum
from nfnets.base import ScaledStdConv2d
from nfnets.models.resnet import NFResNetCIFAR, BasicBlock
from nfnets.models.resnet import activation_fn
from rtsutils.nb_util import NoOpResults
from rtsutils.optim import grads_all_finite
from rtsutils.util import np_to_th, th_to_np
import logging
log = logging.getLogger(__name__)


def run_exp(
    output_dir,
    n_epochs,
    optim_type,
    n_start_filters,
    residual_preproc,
    model_name,
    lr_preproc,
    lr_clf,
    threshold,
    bpd_weight,
    np_th_seed,
    first_n,
    batch_size,
    weight_decay,
    debug,
):
    assert model_name in ["wide_nf_net", "nf_net"]
    tqdm = lambda x: x
    trange = range
    set_random_seeds(np_th_seed, True)
    zero_init_residual = False  # True#False
    initialization = "xavier_normal"  # kaiming_normal
    split_test_off_train = False
    n_warmup_epochs = 0
    bias_for_conv = True
    resample_augmentation = True
    adjust_betas = True

    writer = SummaryWriter(output_dir)

    writer.flush()
    log.info("Load data...")
    data_path = "/home/schirrmr/data/pytorch-datasets/data/CIFAR10/"
    dataset = "CIFAR10"
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

    log.info("Create classifier...")
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    normalize = kornia.augmentation.Normalize(
        mean=np_to_th(mean, device="cpu", dtype=np.float32),
        std=np_to_th(std, device="cpu", dtype=np.float32),
    )
    if model_name == "wide_nf_net":
        depth = 28
        widen_factor = 10
        dropout = 0.3
        from wide_resnet.networks.wide_nfnet import conv_init, Wide_NFResNet

        nf_net = Wide_NFResNet(depth, widen_factor, dropout, num_classes).cuda()
        nf_net.apply(conv_init)
    else:
        assert model_name == "nf_net"
        nf_net = NFResNetCIFAR(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            activation="elu",
            alpha=0.2,
            beta=1.0,
            zero_init_residual=zero_init_residual,
            base_conv=ScaledStdConv2d,
            n_start_filters=n_start_filters,
            bias_for_conv=bias_for_conv,
        )
    clf = nn.Sequential(normalize, nf_net)
    clf = clf.cuda()


    log.info("Create preprocessor...")
    def to_plus_minus_one(x):
        return (x * 2) - 1

    if residual_preproc:

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
    else:

        preproc = nn.Sequential(
            Expression(to_plus_minus_one),
            UnetGenerator(
                3,
                3,
                num_downs=5,
                final_nonlin=nn.Sigmoid,
                norm_layer=AffineOnChans,
                nonlin_down=nn.ELU,
                nonlin_up=nn.ELU,
            ),
            AffineOnChans(3),
        ).cuda()
        preproc[2].factors.data[:] = 0.1


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

    if optim_type == "adam":
        opt_clf = torch.optim.Adam(
            [
                dict(params=params_with_weight_decay, weight_decay=weight_decay),
                dict(params=params_without_weight_decay, weight_decay=0),
            ],
            lr=lr_clf,
            betas=beta_clf,
        )
    else:
        assert optim_type == "adamw"
        opt_clf = torch.optim.AdamW(
            [
                dict(params=params_with_weight_decay, weight_decay=weight_decay),
                dict(params=params_without_weight_decay, weight_decay=0),
            ],
            lr=lr_clf,
            betas=beta_clf,
        )

    opt_preproc = torch.optim.Adam(
        preproc.parameters(), lr=lr_preproc, weight_decay=5e-5, betas=beta_preproc
    )

    for name, module in clf.named_modules():
        if "Conv2d" in module.__class__.__name__:
            assert hasattr(module, "weight")
            nn.init.__dict__[initialization + "_"](module.weight)
        if hasattr(module, "bias"):
            module.bias.data.zero_()

    def get_aug_m(X_shape):
        noise = th.randn(*X_shape, device="cuda") * 1e-3
        aug_m = nn.Sequential(
            FixedAugment(
                kornia.augmentation.RandomCrop(
                    (32, 32),
                    padding=4,
                ),
                X_shape,
            ),
            FixedAugment(kornia.augmentation.RandomHorizontalFlip(), X_shape),
            Expression(lambda x: x + noise),
        )
        return aug_m

    # Adjust betas to unit variance
    if adjust_betas:
        # actually should et the zero init blcoks to 1 before, and afterwards to zero agian... according to paper logic
        for name, module in clf.named_modules():
            if module.__class__.__name__ == "ScalarMultiply":
                module.scalar.data[:] = 1
        with th.no_grad():
            init_X = th.cat(
                [get_aug_m(X.shape)(X.cuda()) for X, y in islice(trainloader, 10)]
            )

        def adjust_beta(module, input):
            assert len(input) == 1
            out = activation_fn[module.activation](input[0])
            std = out.std(dim=(1, 2, 3)).mean()

            assert hasattr(module, "beta")
            module.beta = 1 / float(std)
            print(std)

        handles = []
        for m in clf.modules():
            if m.__class__.__name__ == "BasicBlock":
                handle = m.register_forward_pre_hook(adjust_beta)
                handles.append(handle)

        try:
            with th.no_grad():
                clf(init_X)
        finally:
            for handle in handles:
                handle.remove()
        for name, module in clf.named_modules():
            if module.__class__.__name__ == "ScalarMultiply":
                module.scalar.data[:] = 0

    log.info("Load generative model...")
    glow = torch.load(
        # "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
        "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/10/10_model.th"
    )

    gen = nn.Sequential(Expression(img_0_1_to_glow_img), glow)

    def get_bpd(gen, X):
        n_dims = np.prod(X.shape[1:])
        _, lp = gen(X)
        bpd = -(lp - np.log(256) * n_dims) / (np.log(2) * n_dims)
        return bpd

    log.info("Start training...")
    nb_res = NoOpResults(0.95)
    for i_epoch in trange(n_epochs + n_warmup_epochs):
        for X, y in tqdm(trainloader):
            X = X.cuda()
            y = y.cuda()
            aug_m = get_aug_m(X.shape)

            simple_X = preproc(X)
            simple_X_aug = aug_m(simple_X)
            X_aug = aug_m(X)

            with higher.innerloop_ctx(clf, opt_clf, copy_initial_weights=True) as (
                f_clf,
                f_opt_clf,
            ):
                f_simple_out = f_clf(simple_X_aug)
                f_simple_loss_before = th.nn.functional.cross_entropy(f_simple_out, y)
                f_opt_clf.step(f_simple_loss_before)

            # could resampe aug_m here if wanted
            if resample_augmentation:
                aug_m = get_aug_m(X.shape)
            f_simple_out = f_clf(simple_X_aug)
            f_simple_loss = th.nn.functional.cross_entropy(f_simple_out, y)

            f_orig_out = f_clf(X_aug)
            f_orig_loss = th.nn.functional.cross_entropy(f_orig_out, y)

            bpd = get_bpd(gen, simple_X)
            bpd_loss = th.mean(bpd)
            # im_loss = weighted_sum(1,1,f_simple_loss, 20, f_orig_loss, 1, bpd_loss)
            bpd_factor = (
                bpd_weight
                * ((1 / threshold) * (threshold - f_orig_loss).clamp(0, 1)).detach()
            )
            im_loss = weighted_sum(
                1, 1, f_simple_loss, 10, f_orig_loss, bpd_factor, bpd_loss
            )

            opt_preproc.zero_grad(set_to_none=True)
            im_loss.backward()
            if th.isfinite(bpd).all().item() and grads_all_finite(opt_preproc):
                opt_preproc.step()
            opt_preproc.zero_grad(set_to_none=True)

            with th.no_grad():
                simple_X = preproc(X)
                simple_X_aug = aug_m(simple_X)

            out = clf(simple_X_aug)
            clf_loss = th.nn.functional.cross_entropy(out, y)
            opt_clf.zero_grad(set_to_none=True)
            clf_loss.backward()
            # Maybe keep grads for analysis?
            opt_clf.step()
            opt_clf.zero_grad(set_to_none=True)
            with th.no_grad():
                im_mse_diff = th.nn.functional.mse_loss(X, simple_X) * 100
                orig_bpd = get_bpd(gen, X)
                bpd_diff = th.mean(bpd - orig_bpd.clamp_max(15))
            nb_res.collect(
                im_mse_diff=im_mse_diff.item(),
                f_simple_loss=f_simple_loss.item(),
                f_simple_loss_before=f_simple_loss_before.item(),
                f_orig_loss=f_orig_loss.item(),
                # merge_weight=preproc.merge_weight.item(), might not exist
                bpd_loss=bpd_loss.item(),
                bpd_diff=bpd_diff.item(),
                im_loss=im_loss.item(),
                clf_loss=clf_loss.item(),
            )
            nb_res.print()
            # with nb_res.output_area('lr'):
            #    print(f"LR: {opt_clf.param_groups[0]['lr']:.2E}")
        nb_res.plot_df()
        with nb_res.output_area("accs_losses"):
            results = {}
            with torch.no_grad():
                for with_preproc in [True, False]:
                    for set_name, loader in (
                        ("train", train_det_loader),
                        ("test", testloader),
                    ):
                        all_preds = []
                        all_ys = []
                        all_losses = []
                        if with_preproc:
                            all_bpds = []
                        for X, y in tqdm(loader):
                            X = X.cuda()
                            y = y.cuda()
                            if with_preproc:
                                X_preproced = preproc(X)
                                bpds = get_bpd(gen, X_preproced)
                                all_bpds.append(th_to_np(bpds))
                                preds = clf(X_preproced)
                            else:
                                preds = clf(X)
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
                        if with_preproc:
                            all_bpds = np.concatenate(all_bpds)
                        acc = np.mean(all_preds.argmax(axis=1) == all_ys)
                        mean_loss = np.mean(all_losses)
                        mean_bpd = np.mean(all_bpds)
                        key = set_name + ["_", "_preproc"][with_preproc]
                        print(f"{key.capitalize()} Acc:  {acc:.1%}")
                        print(f"{key.capitalize()} Loss: {mean_loss: .2f}")
                        if with_preproc:
                            print(f"{key.capitalize()} BPD: {mean_bpd: .2f}")
                        writer.add_scalar(key + "_acc", acc * 100, i_epoch)
                        writer.add_scalar(key + "_loss", mean_loss, i_epoch)
                        if with_preproc:
                            writer.add_scalar(key + "_bpd", mean_bpd, i_epoch)
                        results[key + "_acc"] = acc * 100
                        results[key + "_loss"] = mean_loss
                        if with_preproc:
                            results[key + "_bpd"] = mean_bpd
                        writer.flush()
                        sys.stdout.flush()
        with nb_res.output_area("images"):
            X, y = next(testloader.__iter__())
            X = X.cuda()
            y = y.cuda()
            X_preproced = preproc(X)
            # im = rescale(create_rgb_image(stack_images_in_rows(th_to_np(X), th_to_np(X_preproced), n_cols=16)), 2)
            # display(im)
    with th.no_grad():
        X, y = next(testloader.__iter__())
        X = X.cuda()
        if with_preproc:
            X_preproced = preproc(X)
        th.save(
            X_preproced.detach().cpu(),
            os.path.join(output_dir, "X_preproced.th"),
        )
        save_image(
            X_preproced,
            os.path.join(output_dir, "X_preproced.png"),
            nrow=int(np.sqrt(len(X_preproced))),
        )

    return results
