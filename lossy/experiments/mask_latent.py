from lossy.datasets import get_dataset
from lossy import wide_nf_net, data_locations
from lossy.glow import load_glow
from lossy.glow_preproc import Wide_NFResNet_Encoder
from copy import deepcopy
from lossy.modules import Expression
from lossy.image_convert import img_0_1_to_glow_img, add_glow_noise
import numpy as np
from torch import nn
import torch as th
from lossy.optim import PercentileGradClip
from rtsutils.plot import stack_images_in_rows
from rtsutils.util import th_to_np, np_to_th
from rtsutils.nb_util import Results
from lossy.util import set_random_seeds
from torchvision.utils import save_image
from tensorboardX.writer import SummaryWriter
import os.path
import sys
import pandas as pd
import logging


log = logging.getLogger(__name__)



def run_exp(
    n_epochs,
    bpd_weight,
    np_th_seed,
    first_n,
    clip_grad_percentile,
    output_dir,
):
    writer = SummaryWriter(output_dir)
    hparams = {k: v for k, v in locals().items() if v is not None}
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    dataset_name = 'CIFAR10'
    data_path = data_locations.pytorch_data
    reverse = False
    split_test_off_train = False
    batch_size = 64
    stripes_factor = 0.3
    bpd_loss_only_if_pred_correct = True
    tqdm = lambda x : x

    set_random_seeds(np_th_seed, True)

    _, _, num_classes, class_names, train_loader, train_det_loader, test_loader = get_dataset(
        dataset_name,
        data_path,
        standardize=False,
        reverse=reverse,
        first_n=first_n,
        split_test_off_train=split_test_off_train,
        batch_size=batch_size,
        stripes_factor=stripes_factor,
    )
    # Load uninitialized clf
    from lossy.classifier import get_classifier_from_folder
    clf = get_classifier_from_folder(
        '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/610',
        load_weights=False).eval()


    noise_before_generator = False
    log.info("Load generative model...")
    affine_glow = load_glow('/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th')
    # now freeze parameters
    for param in affine_glow.parameters():
        param.requires_grad = False
    affine_gen = nn.Sequential()
    affine_gen.add_module("to_glow_range", Expression(img_0_1_to_glow_img))
    if noise_before_generator:
        affine_gen.add_module("add_noise", Expression(add_glow_noise))
    affine_gen.add_module("glow", affine_glow)

    def get_bpd(gen, X):
        n_dims = np.prod(X.shape[1:])
        _, lp = gen(X)
        bpd = -(lp - np.log(256) * n_dims) / (np.log(2) * n_dims)
        return bpd

    opt_clf = th.optim.AdamW(clf.parameters(), lr=1e-4, weight_decay=5e-5)
    opt_clf = PercentileGradClip(opt_clf, clip_grad_percentile, 400, 10)

    cat_clf_chans = False
    merge_weight_clf_chans = False
    masker = Wide_NFResNet_Encoder(
        16,
        4,
        0,
        "elu",
        cat_clf_chans=cat_clf_chans,
        merge_weight_clf_chans=merge_weight_clf_chans,
        output_chans=[6, 12, 48],
    ).cuda()

    optim_masker = th.optim.AdamW(
        masker.parameters(),
        lr=1e-4,
        weight_decay=5e-5,
        betas=(0.5, 0.99))
    optim_masker = PercentileGradClip(optim_masker, clip_grad_percentile, 400, 10)

    nb_res = Results(0.95)
    for i_epoch in range(n_epochs):
        for i_batch, (X, y) in enumerate(tqdm(train_loader)):
            X = X.cuda()
            y = y.cuda()
            with th.no_grad():
                affine_z = affine_gen(X)[0]
            alpha_masks = masker(X)[0]
            masks = [th.sigmoid(a) for a in alpha_masks]
            masked_z = [m * z for m, z in zip(masks, affine_z)]
            masked_noise = [(1 - m) * th.randn_like(m) for m in masks]
            mixed_z = [m_z + m_n for m_z, m_n in zip(masked_z, masked_noise)]
            inved_mixed = affine_glow.invert(mixed_z)[0] + 0.5
            out_mixed = clf(inved_mixed)
            clf_loss_mixed = th.nn.functional.cross_entropy(out_mixed, y)
            inved_masked = affine_glow.invert(masked_z)[0] + 0.5
            out_masked = clf(inved_masked)
            clf_loss_masked = th.nn.functional.cross_entropy(out_masked, y)
            out_orig = clf(X)
            clf_loss_orig = th.nn.functional.cross_entropy(out_orig, y)
            clf_loss = (clf_loss_masked + clf_loss_mixed + clf_loss_orig) / 3

            bpds = get_bpd(affine_gen, inved_masked + th.rand_like(inved_masked) * (1 / 256.0))
            if bpd_loss_only_if_pred_correct:
                all_preds_correct = (out_mixed.argmax(dim=1) == y) & (out_masked.argmax(dim=1) == y) & (
                            out_orig.argmax(dim=1) == y)
                bpd_factors = all_preds_correct * 1.0
            else:
                bpd_factors = th.zeros_like(bpds)

            nll_loss = th.mean(bpds * bpd_factors)
            loss = clf_loss + nll_loss * bpd_weight
            optim_masker.zero_grad()
            opt_clf.zero_grad()
            loss.backward()
            optim_masker.step()
            opt_clf.step()
            optim_masker.zero_grad()
            opt_clf.zero_grad()
            nb_res.collect(
                loss=loss.item(),
                clf_loss=clf_loss.item(),
                clf_loss_mixed=clf_loss_mixed.item(),
                clf_loss_masked=clf_loss_masked.item(),
                clf_loss_orig=clf_loss_orig.item(),
                nll_loss=nll_loss.item(),
                avg_bpd=th.mean(bpds).item(),
            )

        print(f"Epoch {i_epoch}")
        results = {}
        for setname, loader in (("train", train_det_loader), ("test", test_loader)):
            res_dicts = []
            for X, y in tqdm(loader):
                with th.no_grad():
                    X = X.cuda()
                    y = y.cuda()
                    affine_z = affine_gen(X)[0]
                    alpha_masks = masker(X)[0]
                    masks = [th.sigmoid(a) for a in alpha_masks]
                    masked_z = [m * z for m, z in zip(masks, affine_z)]
                    masked_noise = [(1 - m) * th.randn_like(m) for m in masks]
                    mixed_z = [m_z + m_n for m_z, m_n in zip(masked_z, masked_noise)]
                    inved_mixed = affine_glow.invert(mixed_z)[0] + 0.5
                    out_mixed = clf(inved_mixed)
                    inved_masked = affine_glow.invert(masked_z)[0] + 0.5
                    out_masked = clf(inved_masked)
                    out_orig = clf(X)
                    res_dicts.append(dict(
                        y=th_to_np(y),
                        pred_labels_orig=th_to_np(out_orig.argmax(dim=1)),
                        pred_labels_masked=th_to_np(out_masked.argmax(dim=1)),
                        pred_labels_mixed=th_to_np(out_masked.argmax(dim=1)),
                    ))

            res_df = pd.concat([pd.DataFrame(d) for d in res_dicts])
            results[setname + '_acc_orig'] = np.mean(res_df.pred_labels_orig == res_df.y)
            results[setname + '_acc_masked'] = np.mean(res_df.pred_labels_masked == res_df.y)
            results[setname + '_acc_mixed'] = np.mean(res_df.pred_labels_mixed == res_df.y)

        for key, val in results.items():
            print(f"{key}:  {val:.1%}")
            writer.add_scalar(key, val * 100, i_epoch)
            writer.flush()
            sys.stdout.flush()
        with th.no_grad():
            X,y = next(test_loader.__iter__())
            X = X.cuda()
            y = y.cuda()
            with th.no_grad():
                affine_z = affine_gen(X)[0]
            alpha_masks = masker(X)[0]
            masks = [th.sigmoid(a) for a in alpha_masks]
            masked_z = [m * z for m, z in zip(masks, affine_z)]
            masked_noise = [(1 - m) * th.randn_like(m) for m in masks]
            mixed_z = [m_z + m_n for m_z, m_n in zip(masked_z, masked_noise)]
            inved_mixed = affine_glow.invert(mixed_z)[0] + 0.5
            inved_masked = affine_glow.invert(masked_z)[0] + 0.5

            X_for_plot = th.flatten(
                np_to_th(
                    stack_images_in_rows(
                        th_to_np(X),
                        th_to_np(inved_masked),
                        th_to_np(inved_mixed),
                        n_cols=int(np.sqrt(len(X))),
                    )
                ),
                start_dim=0,
                end_dim=1,
            )

            save_image(
                X_for_plot,
                os.path.join(output_dir, "X_preproced.png"),
                nrow=int(np.sqrt(len(X))),
            )
            th.save(clf.state_dict(), os.path.join(output_dir, "clf_state_dict.th"))
            th.save(masker.state_dict(), os.path.join(output_dir, "masker_state_dict.th"))
    return results

