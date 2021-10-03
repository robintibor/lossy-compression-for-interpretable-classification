import json
import logging
import os.path
import sys

import kornia
import numpy as np
import pandas as pd
import torch
import torch as th
from braindecode.models.modules import Expression
from braindecode.util import set_random_seeds
from tensorboardX.writer import SummaryWriter
from torch import nn

from lossy.affine import AffineOnChans
from lossy.augment import FixedAugment
from lossy.datasets import get_dataset
from lossy.image2image import WrapResidualIdentityUnet, UnetGenerator
from lossy.image_convert import add_glow_noise_to_0_1, img_0_1_to_glow_img
from lossy.image_convert import to_plus_minus_one
from rtsutils.nb_util import NoOpResults
from rtsutils.plot import stack_images_in_rows, rescale, create_rgb_image
from rtsutils.util import np_to_th, th_to_np
from lossy.losses import kl_divergence
from lossy.util import inverse_sigmoid
from lossy.util import weighted_sum
from lossy.optim import set_grads_to_none
from lossy.grad_alignment import cos_sim_neg_grads, mse_neg_grads
from lossy.util import soft_clip

log = logging.getLogger(__name__)


def run_exp(
    output_dir, i_start, images_to_analyze, np_th_seed, n_epochs, bpd_weight, debug
):
    writer = SummaryWriter(output_dir)
    hparams = {k: v for k, v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    tqdm = lambda x: x
    trange = range

    set_random_seeds(np_th_seed, True)
    from lossy.datasets import get_dataset

    (
        channel,
        im_size,
        num_classes,
        class_names,
        trainloader,
        train_det_loader,
        testloader,
    ) = get_dataset("CIFAR10", "/home/schirrmr/pytorch-datasets/", standardize=False)

    import wide_resnet.config as cf

    log.info("Create classifier...")
    dataset = "cifar10"
    mean = cf.mean[dataset]
    std = cf.mean[dataset]  # bug to fix at some point...
    normalize = kornia.augmentation.Normalize(
        mean=np_to_th(mean, device="cpu", dtype=np.float32),
        std=np_to_th(std, device="cpu", dtype=np.float32),
    )

    dropout = 0.3
    activation = "elu"  # was elu in past?
    saved_model_folder = (
        "/work/dlclarge2/schirrmr-lossy-compression/exps/one-step-noise-fixed/271/"
    )
    if saved_model_folder is not None:
        saved_model_config = json.load(
            open(os.path.join(saved_model_folder, "config.json"), "r")
        )
        depth = saved_model_config["depth"]
        widen_factor = saved_model_config["widen_factor"]
        dropout = saved_model_config.get("dropout", 0.3)
        activation = saved_model_config.get(
            "activation", "elu"
        )  # default was relu.. or elu?
        assert saved_model_config.get("dataset", "cifar10") == dataset
    from wide_resnet.networks.wide_nfnet import conv_init, Wide_NFResNet

    nf_net = Wide_NFResNet(
        depth, widen_factor, dropout, num_classes, activation=activation
    ).cuda()
    nf_net.apply(conv_init)
    clf = nn.Sequential(normalize, nf_net)
    clf = clf.cuda()
    if saved_model_folder is not None:
        saved_clf_state_dict = th.load(
            os.path.join(saved_model_folder, "clf_state_dict.th")
        )
        clf.load_state_dict(saved_clf_state_dict)

    clf.eval()
    log.info("Load generative model...")
    glow = torch.load(
        # "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
        "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
    )

    gen = nn.Sequential(Expression(img_0_1_to_glow_img), glow)

    def get_bpd(gen, X):
        n_dims = np.prod(X.shape[1:])
        _, lp = gen(X)
        bpd = -(lp - np.log(256) * n_dims) / (np.log(2) * n_dims)
        return bpd

    from itertools import islice

    X_y_pred = dict(X=[], y=[], pred=[], prob=[])
    for X, y in testloader:
        X_y_pred["X"].extend(X)
        X_y_pred["y"].extend(y)
        X = X.cuda()
        y = y.cuda()
        with th.no_grad():
            out_orig = clf(X)
        X_y_pred["pred"].extend(th_to_np(out_orig))
        X_y_pred["prob"].extend(th_to_np(th.softmax(out_orig, dim=1)))
    for key in ["pred", "prob"]:
        X_y_pred[key] = np.array(X_y_pred[key])
    X_y_pred["X"] = th_to_np(th.stack(X_y_pred["X"]))
    X_y_pred["y"] = th_to_np(th.stack(X_y_pred["y"]))
    print((X_y_pred["pred"].argmax(axis=1) == X_y_pred["y"]).mean())

    mask_correct = X_y_pred["prob"].argmax(axis=1) == X_y_pred["y"]
    mask_incorrect = ~mask_correct

    mask_confident = X_y_pred["prob"].max(axis=1) > 0.99

    if images_to_analyze == "false_pred":
        mask = mask_confident & mask_incorrect
    elif images_to_analyze == "true_pred":
        mask = mask_confident & mask_correct

    else:
        assert False
    print(np.sum(mask))
    batch_size = 32
    # save the indices of the images
    np.save(
        os.path.join(output_dir, "X_inds.npy"),
        np.flatnonzero(mask)[i_start : i_start + batch_size],
    )
    X = th.tensor(X_y_pred["X"][mask])[i_start : i_start + batch_size]
    y = th.tensor(X_y_pred["y"][mask])[i_start : i_start + batch_size]
    X = X.cuda()
    y = y.cuda()
    with th.no_grad():
        orig_out = clf(X)
    labels = [testloader.dataset.classes[a_y] for a_y in y]

    pred_labels = [testloader.dataset.classes[a_p] for a_p in orig_out.argmax(dim=1)]
    glow.remove_cur_in_out()
    orig_params = [p.detach().clone() for p in clf.parameters()]

    with th.no_grad():
        z_orig_X = gen(X)[0]
        z_zero = gen(th.zeros_like(X))[0]

    z_alpha_X = [th.zeros_like(z, requires_grad=True) for z in z_orig_X]
    opt_simple_X_alpha = th.optim.Adam(z_alpha_X, lr=1e-2)

    with th.no_grad():
        simple_X = glow.invert(z_alpha_X)[0] + 0.5
    simple_scaled_target_name = "simple_out"

    weight_unscaled_kl_div = 1
    weight_scaled_kl_div = 1
    mse_weight = 0
    from backpack import extend
    from backpack import backpack
    from lossy.batch_grad import BatchGradNFNets
    from copy import deepcopy

    scaled_clf = deepcopy(clf)
    scaled_clf = extend(scaled_clf)

    nb_res = NoOpResults(0.95)
    # clf.train()
    clf.eval()
    for i_epoch in trange(n_epochs):
        print(f"Epoch {i_epoch:d}")
        glow.remove_cur_in_out()

        set_grads_to_none(z_alpha_X)
        set_grads_to_none(scaled_clf.parameters())
        scaled_clf.eval()

        lower_bound = 0.8
        upper_bound = 0.95
        scale_factor = (th.rand(1) * (upper_bound - lower_bound) + lower_bound).item()
        for p_scaled, p_orig in zip(scaled_clf.parameters(), orig_params):
            p_scaled.data[:] = p_orig.clone().data[:] * scale_factor

        out_scaled_orig = scaled_clf(X)
        kl_div = kl_divergence(orig_out, out_scaled_orig)

        with backpack(BatchGradNFNets()):
            kl_div.backward()
        gate_grads_orig = [
            p.detach().unsqueeze(0) * p.grad_batch.clone().detach()
            for p in scaled_clf.parameters()
        ]

        set_grads_to_none(scaled_clf.parameters())
        # maybe unnecessary
        set_grads_to_none(z_alpha_X)
        set_grads_to_none(glow.parameters())

        interp = 1 - th.rand(1).item()  # * 0.5
        z_interped = [
            z_simple * interp + z_orig * (1 - interp)
            for z_simple, z_orig in zip(z_alpha_X, z_orig_X)
        ]
        simple_X = glow.invert(z_interped)[0] * (255 / 256.0) + 0.5

        if weight_unscaled_kl_div > 0 or simple_scaled_target_name == "simple_out":
            simple_unscaled_out = clf(simple_X)
        assert simple_scaled_target_name in ["simple_out", "orig_out"]
        if simple_scaled_target_name == "simple_out":
            simple_scaled_target = simple_unscaled_out
        else:
            simple_scaled_target = orig_out

        out_scaled_simple = scaled_clf(simple_X)

        kl_div = kl_divergence(simple_scaled_target, out_scaled_simple)
        with backpack(BatchGradNFNets()):
            kl_div.backward(create_graph=True)

        set_grads_to_none(glow.parameters())
        set_grads_to_none(z_alpha_X)

        gate_grads_simple = [
            p.detach().unsqueeze(0) * p.grad_batch for p in scaled_clf.parameters()
        ]
        # now reget this for bpd
        simple_X_glow = glow.invert(z_alpha_X)[0] * (255 / 256.0) + 0.5
        bpds = get_bpd(
            gen,
            soft_clip(simple_X_glow + th.rand_like(simple_X_glow) * 1 / 256.0, 0, 1),
        )
        bpd_loss = th.mean(bpds)
        mse = th.mean(
            th.stack(
                [
                    mse_neg_grads(g_orig, g_simple)
                    for g_orig, g_simple in zip(gate_grads_orig, gate_grads_simple)
                ]
            )
        )
        cos_dist = 1 - th.mean(
            th.stack(
                [
                    cos_sim_neg_grads(g_orig, g_simple)
                    for g_orig, g_simple in zip(gate_grads_orig, gate_grads_simple)
                ]
            )
        )

        kl_div_scaled_orig_simple = kl_divergence(
            out_scaled_orig.detach(), out_scaled_simple
        )

        if weight_unscaled_kl_div > 0:
            kl_div_unscaled_orig_simple = kl_divergence(orig_out, simple_unscaled_out)
        else:
            kl_div_unscaled_orig_simple = th.zeros(1, device=orig_out.device)[0]
        loss = weighted_sum(
            1,
            10,
            cos_dist,
            mse_weight,
            mse,
            bpd_weight,
            bpd_loss,
            weight_scaled_kl_div,
            kl_div_scaled_orig_simple,
            weight_unscaled_kl_div,
            kl_div_unscaled_orig_simple,
        )
        opt_simple_X_alpha.zero_grad(set_to_none=True)
        loss.backward()
        nan_grads = 0
        for z in z_alpha_X:
            for i_ex in range(len(z)):
                if not th.all(th.isfinite(z.grad[i_ex])):
                    z.grad[i_ex] = th.zeros_like(z.grad[i_ex])
                    nan_grads = 1
        opt_simple_X_alpha.step()
        opt_simple_X_alpha.zero_grad(set_to_none=True)

        set_grads_to_none(glow.parameters())
        set_grads_to_none(clf.parameters())
        set_grads_to_none(scaled_clf.parameters())
        set_grads_to_none(z_alpha_X)

        nb_res.collect(
            loss=loss.item(),
            bpd_loss=bpd_loss.item(),
            scale_factor=scale_factor,
            cos_dist=cos_dist.item(),
            kl_div_scaled_orig_simple=kl_div_scaled_orig_simple.item(),
            kl_div_unscaled_orig_simple=kl_div_unscaled_orig_simple.item(),
            nan_grads=nan_grads,
            mse=mse.item() * 1e5,
        )
        nb_res.print()

        if (i_epoch % 50 == 0) or (i_epoch == (n_epochs - 1)):
            nb_res.plot_df()
            with nb_res.output_area("images"):
                with th.no_grad():
                    simple_X = glow.invert(z_alpha_X)[0] + 0.5
                im_arr = stack_images_in_rows(
                    th_to_np(X), th_to_np(simple_X), n_cols=min(len(X), 8)
                )
                padded_im_arr = np.pad(
                    im_arr,
                    ((0, 0), (0, 0), (0, 0), (8, 0), (0, 0)),
                    constant_values=0.8,
                )
                im = rescale(create_rgb_image(padded_im_arr), 3)

                import PIL.ImageDraw as ImageDraw
                from PIL import ImageFont

                draw = ImageDraw.Draw(im)
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
                )
                for i_example, (label, pred_label) in enumerate(
                    zip(labels, pred_labels)
                ):
                    i_col = i_example % 8
                    i_row = i_example // 8
                    draw.text(
                        (
                            i_col * 32 * 3,
                            i_row * (32 + 8) * 3 * 2 + 2,
                        ),
                        label.replace("_", " ").capitalize(),
                        (0, 0, 0),
                        font=font,
                    )
                    draw.text(
                        (
                            i_col * 32 * 3,
                            i_row * (32 + 8) * 3 * 2 + (32 + 8) * 3 + 2,
                        ),
                        pred_label.replace("_", " ").capitalize(),
                        (0, 0, 0),
                        font=font,
                    )
                im.save(os.path.join(output_dir, "pred_image.png"))
            th.save(z_alpha_X, os.path.join(output_dir, "z_alpha.th"))
            th.save(simple_X_glow, os.path.join(output_dir, "simple_X_glow.th"))
    results = dict(
        loss=loss.item(),
        bpd_loss=bpd_loss.item(),
        cos_dist=cos_dist.item(),
        kl_div_scaled_orig_simple=kl_div_scaled_orig_simple.item(),
        kl_div_unscaled_orig_simple=kl_div_unscaled_orig_simple.item(),
        nan_grads=nan_grads,
        mse=mse.item() * 1e5,
    )
    return results
