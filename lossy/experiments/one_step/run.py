import itertools
import json
import logging
import os.path
import sys
import functools
from functools import partial
from itertools import islice
from copy import deepcopy

import higher
import kornia
import numpy as np
import torch
import torch as th
from tensorboardX.writer import SummaryWriter
from torch import nn
from torchvision.utils import save_image

from lossy import wide_nf_net, data_locations
from lossy.activation_match import add_conv_bias_grad
from lossy.activation_match import compute_dist
from lossy.activation_match import conv_weight_grad_loop
from lossy.activation_match import cos_dist_unfolded_grads
from lossy.activation_match import cosine_distance
from lossy.activation_match import detach_acts_grads
from lossy.activation_match import get_in_out_activations_per_module
from lossy.activation_match import get_in_out_acts_and_in_out_grads_per_module
from lossy.activation_match import grad_in_act_act
from lossy.activation_match import gradparam_param
from lossy.activation_match import refed
from lossy.activation_match import unfolded_grads
from lossy.affine import AffineOnChans
from lossy.augment import FixedAugment, TrivialAugmentPerImage
from lossy.condensation.networks import ConvNet
from lossy.datasets import get_dataset
from lossy.glow import load_small_glow
from lossy.image2image import WrapResidualIdentityUnet, UnetGenerator
from lossy.image_convert import add_glow_noise, add_glow_noise_to_0_1
from lossy.image_convert import img_0_1_to_glow_img, quantize_data
from lossy.losses import grad_normed_loss
from lossy.losses import kl_divergence
from lossy.modules import Expression
from lossy.optim import grads_all_finite
from lossy.optim import set_grads_to_none
from lossy.plot import stack_images_in_rows
from lossy.simclr import compute_nt_xent_loss, modified_simclr_pipeline_transform
from lossy.util import np_to_th, th_to_np
from lossy.util import set_random_seeds
from lossy.util import weighted_sum
from lossy.wide_nf_net import activation_fn

log = logging.getLogger(__name__)


def restore_grads_from(params, fieldname):
    for p in params:
        p.grad = getattr(p, fieldname)  # .detach().clone()
        delattr(p, fieldname)


def save_grads_to(params, fieldname):
    for p in params:
        setattr(p, fieldname, p.grad.detach().clone())


def get_clf_and_optim(
    model_name,
    num_classes,
    normalize,
    optim_type,
    saved_model_folder,
    depth,
    widen_factor,
    lr_clf,
    weight_decay,
    activation,
    dataset,
):
    if model_name in ["wide_nf_net", "wide_bnorm_net"]:
        dropout = 0.3
        if saved_model_folder is not None:
            saved_model_config = json.load(
                open(os.path.join(saved_model_folder, "config.json"), "r")
            )
            depth = saved_model_config["depth"]
            widen_factor = saved_model_config["widen_factor"]
            dropout = saved_model_config["dropout"]
            activation = saved_model_config.get(
                "activation", "relu"
            )  # default was relu
            assert saved_model_config.get("dataset", "cifar10") == dataset
        if model_name == "wide_nf_net":
            from lossy.wide_nf_net import conv_init, Wide_NFResNet

            nf_net = Wide_NFResNet(
                depth, widen_factor, dropout, num_classes, activation=activation
            ).cuda()
            nf_net.apply(conv_init)
            model = nf_net
        elif model_name == "wide_bnorm_net":
            assert activation == "relu"
            # activation = "relu"  # overwrite for wide resnet for now
            from lossy.wide_resnet import Wide_ResNet, conv_init

            model = Wide_ResNet(
                depth, widen_factor, dropout, num_classes, activation=activation
            ).cuda()
            model.apply(conv_init)
        else:
            assert False
        if saved_model_folder is not None:
            saved_clf_state_dict = th.load(
                os.path.join(saved_model_folder, "nf_net_state_dict.th")
            )
            model.load_state_dict(saved_clf_state_dict)
    elif model_name == "resnet18":
        from lossy.resnet import resnet18

        model = resnet18(num_classes=num_classes)
    elif model_name == "conv_net":
        model = ConvNet(
            channel=3,
            num_classes=num_classes,
            net_width=64,
            net_depth=3,
            net_act="shifted_softplus",
            net_norm="batchnorm",
            net_pooling="avgpooling",
            im_size=(32, 32),
        )
    else:
        assert False

    clf = nn.Sequential(normalize, model)
    clf = clf.cuda()
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
    return clf, opt_clf


def adjust_betas_of_clf(clf, trainloader, get_aug_m):
    # actually should et the zero init blcoks to 1 before, and afterwards to zero agian... according to paper logic
    for name, module in clf.named_modules():
        if module.__class__.__name__ == "ScalarMultiply":
            module.scalar.data[:] = 1
    with th.no_grad():
        init_X = th.cat(
            [
                get_aug_m(
                    X.shape,
                )(X.cuda())
                for X, y in islice(trainloader, 10)
            ]
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
    adjust_betas,
    saved_model_folder,
    save_models,
    train_orig,
    dataset,
    noise_before_generator,
    noise_after_simplifier,
    noise_augment_level,
    depth,
    widen_factor,
    trivial_augment,
    resample_augmentation,
    resample_augmentation_for_clf,
    std_aug_magnitude,
    extra_augs,
    quantize_after_simplifier,
    train_simclr_orig,
    ssl_loss_factor,
    train_ssl_orig_simple,
    activation,
    loss_name,
    grad_from_orig,
    mimic_cxr_target,
    use_normed_loss,
    separate_orig_clf,
    simple_orig_pred_loss_weight,
    scale_dists_loss_by_n_vals,
):
    assert model_name in ["wide_nf_net", "nf_net", "wide_bnorm_net", "resnet18"]
    if saved_model_folder is not None:
        assert model_name == "wide_nf_net"
    writer = SummaryWriter(output_dir)
    hparams = {k: v for k, v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    tqdm = lambda x: x
    trange = range
    set_random_seeds(np_th_seed, True)
    zero_init_residual = False  # True#False
    initialization = "xavier_normal"  # kaiming_normal
    split_test_off_train = False
    n_warmup_epochs = 0
    bias_for_conv = True

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
        mimic_cxr_target=mimic_cxr_target,
    )

    log.info("Create classifier...")
    mean = wide_nf_net.mean[dataset]
    std = wide_nf_net.std[dataset]

    normalize = kornia.augmentation.Normalize(
        mean=np_to_th(mean, device="cpu", dtype=np.float32),
        std=np_to_th(std, device="cpu", dtype=np.float32),
    )

    clf, opt_clf = get_clf_and_optim(
        model_name,
        num_classes,
        normalize,
        optim_type,
        saved_model_folder,
        depth,
        widen_factor,
        lr_clf,
        weight_decay,
        activation,
        dataset,
    )

    if separate_orig_clf:
        orig_clf, opt_orig_clf = deepcopy((clf, opt_clf))

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
            AffineOnChans(3),  # Does this even make sense after sigmoid?
        ).cuda()
    preproc_post = nn.Sequential()
    if quantize_after_simplifier:
        preproc_post.add_module("quantize", Expression(quantize_data))
    if noise_after_simplifier:
        preproc_post.add_module("add_glow_noise", Expression(add_glow_noise_to_0_1))
    preproc = nn.Sequential(preproc, preproc_post)

    beta_preproc = (0.5, 0.99)

    opt_preproc = torch.optim.Adam(
        preproc.parameters(), lr=lr_preproc, weight_decay=5e-5, betas=beta_preproc
    )

    def get_aug_m(X_shape, trivial_augment, std_aug_magnitude, extra_augs):
        noise = th.randn(*X_shape, device="cuda") * noise_augment_level
        aug_m = nn.Sequential()
        if trivial_augment:
            aug_m.add_module(
                "trivial_augment",
                TrivialAugmentPerImage(
                    X_shape[0],
                    num_magnitude_bins=31,
                    std_aug_magnitude=std_aug_magnitude,
                    extra_augs=extra_augs,
                    same_across_batch=False,
                ),
            )
        else:
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
            assert dataset in [
                "mnist",
                "fashionmnist",
                "svhn",
                "mimic-cxr",
                "stripes",
                "mnist_fashion",
                "mnist_cifar",
            ]
        aug_m.add_module("noise", Expression(lambda x: x + noise))

        return aug_m

    # Adjust betas to unit variance
    if adjust_betas and (saved_model_folder is None):
        adjust_betas_of_clf(
            clf,
            trainloader,
            functools.partial(
                get_aug_m,
                trivial_augment=trivial_augment,
                std_aug_magnitude=std_aug_magnitude,
                extra_augs=extra_augs,
            ),
        )

    log.info("Load generative model...")
    glow = load_small_glow()

    gen = nn.Sequential()
    gen.add_module("to_glow_range", Expression(img_0_1_to_glow_img))
    if noise_before_generator:
        gen.add_module("add_noise", Expression(add_glow_noise))
    gen.add_module("glow", glow)

    def get_bpd(gen, X):
        n_dims = np.prod(X.shape[1:])
        _, lp = gen(X)
        bpd = -(lp - np.log(256) * n_dims) / (np.log(2) * n_dims)
        return bpd

    print("clf", clf)
    log.info("Start training...")
    if loss_name == "grad_act_match":
        use_parameter_counts = True
        val_fn = grad_in_act_act
        if grad_from_orig:
            val_fn = refed(val_fn, "in_grad")
        dist_fn = partial(cosine_distance, eps=1e-15)
        flatten_before_dist = True
    elif loss_name == "unfolded_grad_match":
        use_parameter_counts = False
        conv_grad_fn = add_conv_bias_grad(conv_weight_grad_loop)
        val_fn = partial(unfolded_grads, conv_grad_fn=conv_grad_fn)
        if grad_from_orig:
            val_fn = refed(val_fn, "out_grad")
        dist_fn = partial(cos_dist_unfolded_grads, eps=1e-10)
        flatten_before_dist = False
    elif loss_name == "gradparam_param":
        use_parameter_counts = False
        val_fn = gradparam_param
        if grad_from_orig:
            val_fn = refed(val_fn, "out_grad")
        dist_fn = partial(cosine_distance, eps=1e-15)
        flatten_before_dist = True

    if loss_name in ["grad_act_match", "unfolded_grad_match", "gradparam_param"]:
        if separate_orig_clf:
            clf_for_comparisons = orig_clf
        else:
            clf_for_comparisons = clf
        wanted_modules = [m for m in clf_for_comparisons.modules() if
                          len(list(m.parameters(recurse=False))) > 0]

    for i_epoch in trange(n_epochs + n_warmup_epochs):
        for X, y in tqdm(trainloader):
            clf.train()
            X = X.cuda()
            X = X.requires_grad_(True)
            y = y.cuda()

            if separate_orig_clf:
                for p_orig, p_simple in zip(
                        orig_clf.parameters(),
                        clf.parameters()):
                    p_orig.data.copy_(p_simple.data.detach())
                orig_out = orig_clf(X)
                orig_clf_loss = th.nn.functional.cross_entropy(orig_out, y)

                opt_orig_clf.zero_grad(set_to_none=True)
                orig_clf_loss.backward()
                opt_orig_clf.step()
                opt_orig_clf.zero_grad(set_to_none=True)

            aug_m = get_aug_m(
                X.shape,
                trivial_augment=trivial_augment,
                std_aug_magnitude=std_aug_magnitude,
                extra_augs=extra_augs,
            )

            simple_X = preproc(X)
            simple_X_aug = aug_m(simple_X)
            X_aug = aug_m(X)

            if loss_name in ["grad_act_match", "unfolded_grad_match",
                             "gradparam_param"]:
                loss_fn = lambda o: th.nn.functional.cross_entropy(
                    o, y, reduction='none')
                if use_normed_loss:
                    loss_fn = grad_normed_loss(loss_fn)
                else:
                    unmeaned_loss_fn = loss_fn
                    loss_fn = lambda o: th.mean(unmeaned_loss_fn(o))

                random_state = torch.get_rng_state()
                orig_acts_grads = get_in_out_acts_and_in_out_grads_per_module(
                    clf_for_comparisons, X_aug, loss_fn, wanted_modules=wanted_modules
                )
                orig_acts_grads = detach_acts_grads(orig_acts_grads)

                set_grads_to_none(clf_for_comparisons.parameters())

                torch.set_rng_state(random_state)
                if grad_from_orig:
                    simple_acts_grads = get_in_out_activations_per_module(
                        clf_for_comparisons,
                        simple_X_aug,
                        wanted_modules=wanted_modules,
                    )
                    if (not separate_orig_clf) and (not use_normed_loss):

                        # Compute and store gradients for classifier update
                        # loss = loss_fn(simple_acts_grads[wanted_modules[-1]]["out_act"][0])
                        loss = loss_fn(list(simple_acts_grads.values())[-1]["out_act"][0])
                        clf_param_grads = th.autograd.grad(
                            loss, clf.parameters(), retain_graph=True
                        )
                        for p, grad in zip(clf.parameters(), clf_param_grads):
                            p.grad = grad
                else:
                    simple_acts_grads = get_in_out_acts_and_in_out_grads_per_module(
                        clf_for_comparisons,
                        simple_X_aug,
                        loss_fn,
                        wanted_modules=wanted_modules,
                        create_graph=True,
                    )

                if (not separate_orig_clf) and (not use_normed_loss):
                    save_grads_to(clf.parameters(), "grad_tmp")
                set_grads_to_none(clf_for_comparisons.parameters())
                set_grads_to_none(clf.parameters())
                set_grads_to_none(preproc.parameters())
                set_grads_to_none([X])

                dists_per_example = compute_dist(
                    dist_fn,
                    val_fn,
                    simple_acts_grads,
                    orig_acts_grads,
                    flatten_before=flatten_before_dist,
                )

                dists = torch.mean(torch.stack(dists_per_example), dim=1)
                if use_parameter_counts:
                    p_counts = [
                        sum([p.numel() for p in m.parameters(recurse=False)])
                        for m in wanted_modules
                    ]
                else:
                    p_counts = p_counts = [1] * len(dists)
                assert len(p_counts) == len(dists)
                dist_loss_weight = [1, len(dists)][scale_dists_loss_by_n_vals]
                dist_loss = weighted_sum(
                    dist_loss_weight, *list(itertools.chain(*list(zip(p_counts, dists))))
                )

                if simple_orig_pred_loss_weight > 0:
                    simple_pred_loss_adjusted_weight = simple_orig_pred_loss_weight / (simple_orig_pred_loss_weight + 1)
                    simple_to_orig_loss = kl_divergence(
                        orig_acts_grads[wanted_modules[-1]]['out_act'][0].detach(),
                        simple_acts_grads[wanted_modules[-1]]['out_act'][0]) * simple_pred_loss_adjusted_weight
                    dist_loss = dist_loss / (simple_orig_pred_loss_weight + 1)
                else:
                    simple_to_orig_loss = th.zeros_like(dist_loss)
                f_simple_loss_before = simple_to_orig_loss
                f_simple_loss = dist_loss
                f_orig_loss = th.zeros_like(f_simple_loss)
                bpd_factors = th.ones_like(X[:, 0, 0, 0])
            else:
                assert loss_name == 'one_step'
                with higher.innerloop_ctx(clf, opt_clf, copy_initial_weights=True) as (
                        f_clf,
                        f_opt_clf,
                ):
                    random_state = torch.get_rng_state()
                    f_simple_out = f_clf(simple_X_aug)
                    f_simple_loss_before = th.nn.functional.cross_entropy(
                        f_simple_out, y
                    )
                    f_opt_clf.step(f_simple_loss_before)

                torch.set_rng_state(random_state)
                f_simple_out = f_clf(simple_X_aug)
                f_simple_loss = th.nn.functional.cross_entropy(f_simple_out, y)

                torch.set_rng_state(random_state)
                f_orig_out = f_clf(X_aug)
                f_orig_loss_per_ex = th.nn.functional.cross_entropy(
                    f_orig_out, y, reduction="none"
                )
                f_orig_loss = th.mean(f_orig_loss_per_ex)
                bpd_factors = (
                        (1 / threshold) * (threshold - f_orig_loss_per_ex).clamp(0, 1)
                ).detach()

            bpd = get_bpd(gen, simple_X)
            bpd_loss = th.mean(bpd * bpd_factors)
            im_loss = weighted_sum(
                1,
                1,
                f_simple_loss_before,
                1,
                f_simple_loss,
                10,
                f_orig_loss,
                bpd_weight,
                bpd_loss,
            )

            opt_preproc.zero_grad(set_to_none=True)
            im_loss.backward()
            if th.isfinite(bpd).all().item() and grads_all_finite(opt_preproc):
                opt_preproc.step()
            opt_preproc.zero_grad(set_to_none=True)

            if loss_name in ["grad_act_match", "unfolded_grad_match", "gradparam_param"] and (
                    not separate_orig_clf) and (not use_normed_loss):
                restore_grads_from(clf.parameters(), "grad_tmp")
            else:
                # Classifier training
                if resample_augmentation_for_clf:
                    aug_m = get_aug_m(
                        X.shape,
                        trivial_augment=trivial_augment,
                        std_aug_magnitude=std_aug_magnitude,
                        extra_augs=extra_augs,
                    )
                with th.no_grad():
                    simple_X = preproc(X)
                    simple_X_aug = aug_m(simple_X)

                torch.set_rng_state(random_state)
                z_simple = clf[1].compute_features(clf[0](simple_X_aug))
                out = clf[1].linear(z_simple)
                clf_loss = th.nn.functional.cross_entropy(out, y)
                if train_orig:
                    out_orig = clf(X_aug.detach())
                    clf_loss_orig = th.nn.functional.cross_entropy(out_orig, y)
                    clf_loss = (clf_loss + clf_loss_orig) / 2

                if train_simclr_orig:
                    simclr_aug = modified_simclr_pipeline_transform(True)
                    X1_X2 = [simclr_aug(x) for x in X]
                    X1 = th.stack([x1 for x1, x2 in X1_X2]).cuda()
                    X2 = th.stack([x2 for x1, x2 in X1_X2]).cuda()
                    z1 = clf[1].compute_features(clf[0](X1))
                    z2 = clf[1].compute_features(clf[0](X2))
                    simclr_loss = compute_nt_xent_loss(z1, z2)
                    clf_loss = (clf_loss + ssl_loss_factor * simclr_loss) / (
                            1 + ssl_loss_factor
                    )

                if train_ssl_orig_simple:
                    z_orig = clf[1].compute_features(clf[0](X_aug.detach()))
                    ssl_loss = compute_nt_xent_loss(z_simple, z_orig)
                    clf_loss = (clf_loss + ssl_loss_factor * ssl_loss) / (
                            1 + ssl_loss_factor
                    )

                opt_clf.zero_grad(set_to_none=True)
                clf_loss.backward()
            opt_clf.step()
            opt_clf.zero_grad(set_to_none=True)
        clf.eval()
        log.info(f"Epoch {i_epoch:d}")
        results = {}
        if train_simclr_orig:
            results["simclr_loss"] = simclr_loss.item()
            writer.add_scalar("simclr_loss", simclr_loss.item(), i_epoch)
        elif train_ssl_orig_simple:
            results["ssl_loss"] = ssl_loss.item()
            writer.add_scalar("ssl_loss", ssl_loss.item(), i_epoch)

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
            X, y = next(testloader.__iter__())
            X = X.cuda()
            y = y.cuda()
            X_preproced = preproc(X)
            th.save(
                X_preproced.detach().cpu(),
                os.path.join(output_dir, "X_preproced.th"),
            )
            X_for_plot = th.flatten(np_to_th(stack_images_in_rows(
                th_to_np(X),
                th_to_np(X_preproced),
                n_cols=int(np.sqrt(len(X_preproced))))),
                start_dim=0, end_dim=1)

            save_image(
                X_for_plot,
                os.path.join(output_dir, "X_preproced.png"),
                nrow=int(np.sqrt(len(X_preproced))),
            )

        if save_models:
            th.save(
                preproc.state_dict(), os.path.join(output_dir, "preproc_state_dict.th")
            )
            th.save(clf.state_dict(), os.path.join(output_dir, "clf_state_dict.th"))
    return results


if __name__ == "__main__":
    dataset = "cifar10"
    saved_model_folder = None

    n_epochs = 100
    batch_size = 32
    train_orig = False
    noise_augment_level = 0
    noise_after_simplifier = True
    noise_before_generator = False
    trivial_augment = True
    extra_augs = True
    np_th_seed = 0
    depth = 16
    widen_factor = 2
    n_start_filters = 64
    residual_preproc = True
    model_name = "wide_nf_net"
    adjust_betas = False
    save_models = True
    resample_augmentation = False
    resample_augmentation_for_clf = False
    std_aug_magnitude = None
    weight_decay = 1e-05
    lr_clf = 0.0005
    lr_preproc = 0.0005
    threshold = 0.1
    optim_type = "adamw"
    bpd_weight = 0.0
    first_n = None
    debug = True
    if debug:
        first_n = 1024
        n_epochs = 3
    output_dir = "."

    run_exp(
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
        adjust_betas,
        saved_model_folder,
        save_models,
        train_orig,
        dataset,
        noise_before_generator,
        noise_after_simplifier,
        noise_augment_level,
        depth,
        widen_factor,
        trivial_augment,
        resample_augmentation,
        resample_augmentation_for_clf,
        std_aug_magnitude,
        extra_augs,
    )
