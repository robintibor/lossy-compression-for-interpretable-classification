import itertools
import json
import logging
import os.path
import sys


import kornia
import numpy as np
import pandas as pd
import torch
import torch as th

from jpg import JPGCompress
from lossy.glow import load_small_glow
from lossy.modules import Expression
from lossy.util import set_random_seeds
from tensorboardX.writer import SummaryWriter
from torch import nn

from lossy import data_locations
from lossy.affine import AffineOnChans
from lossy.augment import FixedAugment
from lossy.datasets import get_dataset
from lossy.glow import load_glow
from lossy.glow_preproc import get_glow_preproc
from lossy.image2image import WrapResidualIdentityUnet, UnetGenerator
from lossy.image_convert import add_glow_noise_to_0_1, quantize_data, ContrastNormalize, img_0_1_to_glow_img
from lossy.image_convert import to_plus_minus_one
from lossy.util import np_to_th, th_to_np
from lossy.classifier import get_classifier_from_folder


from kornia.filters import GaussianBlur2d
from lossy.simclr import compute_nt_xent_loss, modified_simclr_pipeline_transform
from lossy.preproc import get_preprocessor_from_folder

log = logging.getLogger(__name__)



def run_exp(
    output_dir,
    first_n,
    saved_preproc_exp_folder,
    n_epochs,
    init_pretrained_clf,
    lr_clf,
    np_th_seed,
    weight_decay,
    debug,
    save_models,
    with_batchnorm,
    restandardize_inputs,
    contrast_normalize,
    add_original_data,
    dataset,
    blur_simplifier,
    blur_sigma,
    jpg_quality,
    simclr_loss_factor,
    use_saved_clf_model_folder,
    saved_clf_exp_folder,
):
    writer = SummaryWriter(output_dir)
    hparams = {k: v for k, v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    tqdm = lambda x: x
    trange = range

    set_random_seeds(np_th_seed, True)

    assert not contrast_normalize
    assert not restandardize_inputs
    assert not with_batchnorm

    if (blur_simplifier) or (jpg_quality is not None):
        assert dataset is not None
        config = {
            'noise_augment_level': 0,
            'saved_model_folder': None,
            'dataset': dataset,
            'batch_size': 32,
            'depth': 16,
            'widen_factor': 2,
        }
    else:
        config = json.load(open(os.path.join(saved_preproc_exp_folder, "config.json"), "r"))
    noise_augment_level = config.get("noise_augment_level", 0)
    dataset = config["dataset"]

    # Ignore for now assert config["noise_after_simplifier"]
    batch_size = config["batch_size"]
    split_test_off_train = False
    mimic_cxr_target = config.get('mimic_cxr_target', None)
    stripes_factor = config.get('stripes_factor', None)

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
        stripes_factor=stripes_factor,
    )

    log.info("Create preprocessor...")
    if blur_simplifier:
        preproc = GaussianBlur2d((15, 15), (blur_sigma, blur_sigma))
    elif jpg_quality is not None:
        preproc = JPGCompress(int(jpg_quality))
    else:
        preproc = get_preprocessor_from_folder(saved_preproc_exp_folder)

    log.info("Create classifier...")
    clf = get_classifier_from_folder(saved_clf_exp_folder,
                                     load_weights=init_pretrained_clf)

    log.info("Create optimizers...")
    params_with_weight_decay = []
    params_without_weight_decay = []
    for name, param in clf.named_parameters():
        if "weight" in name or "gain" in name or "cls_token" in name or "pos_emb" in name:
            params_with_weight_decay.append(param)
        else:
            assert "bias" in name, f"Unknown parameter name {name}"
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
            assert dataset in ["mnist", "fashionmnist", "svhn", "mimic-cxr"]
        aug_m.add_module("noise", Expression(lambda x: x + noise))

        return aug_m

    # Construct fixed train/test sets

    def get_tensor_loaders(loader, preproc, batch_size, add_original_data):
        all_X = []
        all_y = []
        all_orig_X = []
        for X, y in tqdm(loader):
            X = X.cuda()
            with th.no_grad():
                simple_X = preproc(X)
            all_X.append(simple_X.cpu())
            all_orig_X.append(X.cpu())
            all_y.append(y.cpu())
        if add_original_data:
            tensor_set = th.utils.data.TensorDataset(
                th.cat(all_X), th.cat(all_orig_X), th.cat(all_y))
        else:
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
        train_det_loader, preproc, batch_size,
        add_original_data=(add_original_data or (simclr_loss_factor is not None)),
    )
    _, testloader_tensors = get_tensor_loaders(testloader, preproc, batch_size,
                                               add_original_data=(
                                                           add_original_data or (simclr_loss_factor is not None)))
    for i_epoch in trange(n_epochs):
        for batch in tqdm(trainloader_tensors):
            if add_original_data or (simclr_loss_factor is not None):
                simple_X, orig_X, y = batch
                orig_X = orig_X.cuda()
            else:
                simple_X, y = batch

            clf.train()
            simple_X = simple_X.cuda()
            y = y.cuda()
            with th.no_grad():
                aug_m = get_aug_m(simple_X.shape, noise_augment_level)
            # since this is how it was trained before
            simple_X = add_glow_noise_to_0_1(simple_X)
            simple_X_aug = aug_m(simple_X)
            out = clf(simple_X_aug)
            clf_loss = th.nn.functional.cross_entropy(out, y)
            if add_original_data:
                orig_out = clf(orig_X)
                orig_clf_loss = th.nn.functional.cross_entropy(orig_out, y)
                clf_loss = (orig_clf_loss + clf_loss) / 2
            if simclr_loss_factor is not None:
                simclr_aug = modified_simclr_pipeline_transform(True)
                X1_X2 = [simclr_aug(x) for x in orig_X]
                X1 = th.stack([x1 for x1, x2 in X1_X2]).cuda()
                X2 = th.stack([x2 for x1, x2 in X1_X2]).cuda()
                z1 = clf[1].compute_features(clf[0](X1))
                z2 = clf[1].compute_features(clf[0](X2))
                simclr_loss = compute_nt_xent_loss(z1, z2)
                clf_loss = (clf_loss + simclr_loss_factor * simclr_loss) / (
                        1 + simclr_loss_factor)

            opt_clf.zero_grad(set_to_none=True)
            clf_loss.backward()
            opt_clf.step()
            opt_clf.zero_grad(set_to_none=True)
        clf.eval()
        results = {}
        print(f"Epoch {i_epoch:d}")
        for loader_name, loader in (
                ("train_preproc", train_det_loader_tensors),
                ("train", train_det_loader),
                ("test_preproc", testloader_tensors),
                ("test", testloader),
        ):
            eval_df = pd.DataFrame()
            for batch in tqdm(loader):
                if (add_original_data or (simclr_loss_factor is not None)) and ("preproc" in loader_name):
                    X, _, y = batch
                else:
                    X, y = batch
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
    if (blur_simplifier) or (jpg_quality is not None):
        log.info("Compute BPOs...")
        log.info("Load generative model...")
        glow = load_small_glow()

        gen = nn.Sequential()
        gen.add_module("to_glow_range", Expression(img_0_1_to_glow_img))
        gen.add_module("glow", glow)

        def get_bpd(gen, X):
            n_dims = np.prod(X.shape[1:])
            _, lp = gen(X)
            bpd = -(lp - np.log(256) * n_dims) / (np.log(2) * n_dims)
            return bpd

        for loader_name, loader in (
                ("train_preproc", train_det_loader_tensors),
                ("test_preproc", testloader_tensors),
        ):
            bpds = []
            for batch in tqdm(loader):
                if (add_original_data or simclr_loss_factor is not None) and ("preproc" in loader_name):
                    X, _, y = batch
                else:
                    X, y = batch
                X = X.cuda()
                y = y.cuda()
                with th.no_grad():
                    simple_X = add_glow_noise_to_0_1(preproc(X))
                    this_bpds = get_bpd(gen, simple_X)
                    bpds.append(th_to_np(this_bpds))
            results[loader_name + '_bpd'] = np.mean(np.concatenate(bpds))
    return results


if __name__ == "__main__":
    n_epochs = 100
    init_pretrained_clf = False
    np_th_seed = 0
    save_models = True
    with_batchnorm = False
    weight_decay = 1e-05
    lr_clf = 0.0005
    first_n = None
    saved_exp_folder = None  # has to be set to model that was saved from one_step/run.py
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
    )
