import os.path
import sys
from itertools import islice

import kornia
import numpy as np
import torch
import torch as th
from lossy.modules import Expression
from lossy.util import set_random_seeds
from tensorboardX.writer import SummaryWriter
from torch import nn
from torchvision.utils import save_image

import higher
from lossy.affine import AffineOnChans
from lossy.augment import FixedAugment, TrivialAugmentPerImage
from lossy.datasets import get_dataset
from lossy.glow import load_small_glow
from lossy.image2image import WrapResidualIdentityUnet, UnetGenerator
from lossy.image_convert import img_0_1_to_glow_img, quantize_data
from lossy.util import weighted_sum
from lossy.optim import grads_all_finite
from lossy.util import np_to_th, th_to_np
import logging
import json
from lossy.image_convert import add_glow_noise, add_glow_noise_to_0_1
from lossy.wide_nf_net import Wide_NFResNet
from lossy.wide_nf_net import activation_fn, conv_init
from lossy import wide_nf_net, data_locations
from rtsutils.nb_util import NoOpResults

log = logging.getLogger(__name__)


def run_exp(
    output_dir,
    n_epochs,
    lr_preproc,
    bpd_weight,
    np_th_seed,
    batch_size,
    first_n,
    dataset,
    save_models,
    noise_before_generator,
    noise_after_simplifier,
    quantize_after_simplifier,
    mse_weight,
):
    writer = SummaryWriter(output_dir)

    hparams = {k: v for k, v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    tqdm = lambda x: x
    trange = range
    set_random_seeds(np_th_seed, True)
    split_test_off_train = False
    n_warmup_epochs = 0
    residual_preproc = True

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
        raise NotImplementedError("No longer implemented")
    preproc_post = nn.Sequential()
    if quantize_after_simplifier:
        preproc_post.add_module("quantize", Expression(quantize_data))
    if noise_after_simplifier:
        preproc_post.add_module("add_glow_noise", Expression(add_glow_noise_to_0_1))
    preproc = nn.Sequential(preproc, preproc_post)
    log.info("Create optimizers...")

    beta_preproc = (0.5, 0.99)
    opt_preproc = torch.optim.Adam(
        preproc.parameters(), lr=lr_preproc, weight_decay=5e-5, betas=beta_preproc
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


    log.info("Start training...")
    nb_res = NoOpResults(0.95)
    for i_epoch in trange(n_epochs + n_warmup_epochs):
        for X, y in tqdm(trainloader):
            preproc.train()
            X = X.cuda()
            y = y.cuda()
            simple_X = preproc(X)
            mse_loss = th.nn.functional.mse_loss(simple_X, X)
            bpd = get_bpd(gen, simple_X)
            bpd_loss = th.mean(bpd)
            im_loss = weighted_sum(
                1,
                mse_weight,
                mse_loss,
                bpd_weight,
                bpd_loss,
            )
            opt_preproc.zero_grad(set_to_none=True)
            im_loss.backward()
            if th.isfinite(bpd).all().item() and grads_all_finite(opt_preproc):
                opt_preproc.step()
            opt_preproc.zero_grad(set_to_none=True)
            nb_res.collect(
                im_loss=im_loss.item(),
                bpd_loss=bpd_loss.item(),
                mse_loss=mse_loss.item() * mse_weight)
            nb_res.print()

        preproc.eval()
        nb_res.plot_df()

        results = {}
        print(f"Epoch {i_epoch:d}")
        with torch.no_grad():
            for set_name, loader in (
                    ("train", train_det_loader),
                    ("test", testloader),
            ):
                all_losses = []
                all_bpds = []
                all_mses = []
                for X, y in tqdm(loader):
                    X = X.cuda()
                    y = y.cuda()
                    X_preproced = preproc(X)
                    bpds = get_bpd(gen, X_preproced)
                    all_bpds.append(th_to_np(bpds))
                    mses = th.nn.functional.mse_loss(X_preproced, X, reduction='none').mean(dim=(1, 2, 3))
                    all_mses.append(th_to_np(mses))
                    all_losses.append(
                        th_to_np(
                            mses * mse_weight + bpds * bpd_weight)
                    )
                all_losses = np.concatenate(all_losses)
                all_bpds = np.concatenate(all_bpds)
                all_mses = np.concatenate(all_mses)
                mean_loss = np.mean(all_losses)
                mean_bpd = np.mean(all_bpds)
                mean_mse = np.mean(all_mses)
                key = set_name
                print(f"{key.capitalize()} BPD:  {mean_bpd:.2f}")
                print(f"{key.capitalize()} Loss: {mean_loss: .2f}")
                print(f"{key.capitalize()} MSE: {mean_mse * 100: .2f}")
                writer.add_scalar(key + "_bpd", mean_bpd, i_epoch)
                writer.add_scalar(key + "_loss", mean_loss, i_epoch)
                writer.add_scalar(key + "_mse", mean_mse * 100, i_epoch)
                results[key + "_loss"] = mean_loss
                results[key + "_bpd"] = mean_bpd
                results[key + "_mse"] = mean_mse * 100
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
            save_image(
                X_preproced,
                os.path.join(output_dir, "X_preproced.png"),
                nrow=int(np.sqrt(len(X_preproced))),
            )
        if save_models:
            th.save(
                preproc.state_dict(), os.path.join(output_dir, "preproc_state_dict.th")
            )

    return results
