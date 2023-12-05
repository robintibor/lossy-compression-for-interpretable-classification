import itertools
import logging
import os.path
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch as th
import torchvision.models as models
from rtsutils.nb_util import Results
from rtsutils.optim import grads_all_finite
from rtsutils.plot import stack_images_in_rows, create_rgb_image
from rtsutils.util import th_to_np, np_to_th
from torch import nn
from torchvision import transforms
from tqdm import tqdm, trange
import PIL.ImageDraw as ImageDraw
from PIL import ImageFont

from lossy.activation_match import cosine_distance, mse_loss, sse_loss
from lossy.activation_match import detach_acts_grads, compute_dist, filter_act_grads
from lossy.activation_match import get_in_out_activations_per_module
from lossy.activation_match import get_in_out_acts_and_in_out_grads_per_module
from lossy.activation_match import grad_in_act_act
from lossy.activation_match import relu_match, refed, relued
from lossy.losses import kl_divergence
from lossy.optim import grads_all_finite
from lossy.optim import set_grads_to_none
from lossy.softplus import ReLUSoftPlusGrad
from lossy.util import set_random_seeds
from lossy.util import weighted_sum

log = logging.getLogger(__name__)

def run_exp(model_name, n_epochs, image_inds, bpd_weight, np_th_seed, debug,  val_fn_name, ref_from_orig, output_dir,
            softplus_beta):
    set_random_seeds(np_th_seed, True)
    from lossy.datasets import ImageNet

    # https://github.com/pytorch/examples/blob/fcf8f9498e40863405fe367b9521269e03d7f521/imagenet/main.py#L213-L237
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    log.info("Loading data...")

    root = "/data/datasets/ImageNet/imagenet-pytorch"

    valid_dataset = ImageNet(
        root=root,
        split="val",
        transform=valid_transform,
        ignore_archive=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    normalize_imagenet = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    log.info("Loading classifier...")
    model = getattr(models, model_name)(pretrained=True).cuda().eval()
    clf = nn.Sequential(normalize_imagenet, model)

    orig_clf = deepcopy(clf)

    for module in clf.modules():
        for key in [
            "relu",
            "relu0",
            "relu1",
        ]:
            if hasattr(module, key):
                setattr(module, key, ReLUSoftPlusGrad(nn.Softplus(beta=softplus_beta)))

    from itertools import islice

    log.info("Computing predictions...")
    preds_df = pd.DataFrame()
    clf.eval()
    for X, y in islice(tqdm(valid_loader), 100):
        with th.no_grad():
            out = clf(X.cuda())

        preds_prob = th_to_np(th.softmax(out, dim=1))
        preds_df = pd.concat(
            (
                preds_df,
                pd.DataFrame(
                    dict(
                        y=y,
                        preds_prob=list(preds_prob),
                        pred_label=preds_prob.argmax(axis=1),
                        max_prob=preds_prob.max(axis=1),
                        X=list(th_to_np(X)),
                    )
                ),
            )
        ).reset_index(drop=True)
    acc = np.mean(preds_df.pred_label == preds_df.y)
    print(acc)


    selected_df = preds_df.loc[image_inds]
    # [[273, 2408]]#[[2968,10]]#[[1356,2030]]#x]#[[2720,2383]]##preds_df.loc[[2968,10]]#,1057,2030,]]#343]]#preds_df.loc[[2720,2383]]
    # [[1603,1346]]frognewt
    # [[2720,2383]] hongose agama
    # [[2968,10]] snake site
    # '1356 eft                  -> bottlecap             (97.7%)',
    # '2030 American chameleon   -> walking stick         (97.2%)',
    # [[1307, 1075]] newteft (and kitebulbar!!)
    selected_X = np_to_th(np.stack(selected_df.X), dtype=np.float32)

    selected_y = torch.tensor(list(selected_df.y))
    selected_pred_y = torch.tensor(list(selected_df.pred_label))

    im_arr = stack_images_in_rows(
        th_to_np(selected_X), th_to_np(selected_X), n_cols=min(len(selected_df), 8)
    )
    padded_im_arr = np.pad(
        im_arr,
        ((0, 0), (0, 0), (0, 0), (24, 0), (0, 0)),
        constant_values=0.8,
    )
    im = create_rgb_image(padded_im_arr)

    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

    labels = [valid_dataset.classes[a_y][0] for a_y in selected_y]
    pred_labels = [valid_dataset.classes[i][0] for i in selected_pred_y]

    # create_rgb_image(th_to_np(selected_X)[None])

    sse_dist_fn = partial(sse_loss)
    mse_dist_fn = partial(mse_loss)
    cos_dist_fn = partial(
        cosine_distance,
        dim=1,
    )

    gen = th.load(
        "/work/dlclarge2/schirrmr-lossy-compression/exps/icml-rebuttal/large-res-glow/8/gen_2.th"
    )

    glow = gen[1]

    def get_bpd(gen, X):
        n_dims = np.prod(X.shape[1:])
        _, lp = gen(X)
        bpd = -(lp - np.log(256) * n_dims) / (np.log(2) * n_dims)
        return bpd

    wanted_modules = []
    wanted_names = []
    for name, module in clf.named_modules():
        if len(list(module.parameters(recurse=False))) > 0:
            wanted_modules.append(module)
            wanted_names.append(name)

    glow.remove_cur_in_out()

    with th.no_grad():
        orig_out = clf(selected_X.cuda())
        pred_y = th.argmax(orig_out, dim=1)

    wanted_y = th.flatten(th.stack((selected_y.cuda(), pred_y), dim=1))

    this_X = th.repeat_interleave(selected_X.cuda(), 2, dim=0).requires_grad_(True)
    this_y = th.repeat_interleave(selected_y.cuda(), 2)

    with th.no_grad():
        orig_out = clf(this_X)
    with th.no_grad():
        z_orig_X = gen(this_X)[0]
        z_grey = gen(th.zeros_like(this_X) + 0.5)[0]

    orig_factor = 0.25  # 1#0.25

    z_alpha_X = [
        (a_z_grey * (1 - orig_factor) + z_orig * orig_factor)
        .clone()
        .requires_grad_(True)
        for a_z_grey, z_orig in zip(z_grey, z_orig_X)
    ]

    opt_simple_X_alpha = th.optim.Adam(
        [
            dict(params=z_alpha_X, lr=1e-2),
        ]
    )

    wanted_label_output_fn = lambda o: th.sum(
        th.stack([a_o[a_y] for a_o, a_y in zip(th.log_softmax(o, dim=1), wanted_y)])
    )
    loss_fn = wanted_label_output_fn

    orig_acts_grads = get_in_out_acts_and_in_out_grads_per_module(
        clf, this_X.cuda(), loss_fn, wanted_modules=wanted_modules
    )

    orig_acts_grads = detach_acts_grads(orig_acts_grads)
    orig_acts_grads = filter_act_grads(orig_acts_grads, ("in_act", "in_grad"))

    wanted_modules = list(orig_acts_grads.keys())

    wanted_names = []
    for wanted_m in wanted_modules:
        for n, m in clf.named_modules():
            if m is wanted_m:
                wanted_names.append(n)

    task_weight = 1  # sse3e-1#mse1e6
    mse_z_weight = 0

    if ref_from_orig:
        val_fn = refed(grad_in_act_act, "in_grad")
    else:
        val_fn = grad_in_act_act

    if val_fn_name == 'relued':
        val_fn = relued(val_fn)
    elif val_fn_name == 'relu_match':
        val_fn = relu_match(val_fn)
    else:
        assert False
    dist_fn = cos_dist_fn  # cos_dist_fn#l1_loss#mse_dist_fn#cos_dist_fn#mse_dist_fn#cos_dist_fn#sse_dist_fn#mse_dist_fn

    def aug_m(x):
        return x * 255 / 256.0 + th.rand_like(x) * (1 / 256.0)

    nb_res = Results(0.95)
    for i_epoch in trange(n_epochs):
        glow.remove_cur_in_out()
        clf.eval()
        simple_X = glow.invert(z_alpha_X)[0] + 0.5
        out_of_0_1_loss = th.nn.functional.relu(th.abs(simple_X) - 1).square().sum()
        simple_X = th.clamp(simple_X, 0, 1)
        # Get only forward acts and then copy into appropriate structure
        if ref_from_orig:
            this_acts = get_in_out_activations_per_module(
                clf,
                aug_m(simple_X),
                wanted_modules=wanted_modules,
            )
        else:
            this_acts = get_in_out_acts_and_in_out_grads_per_module(
                clf, aug_m(simple_X), loss_fn, wanted_modules=wanted_modules,
                create_graph=True,
            )

        set_grads_to_none(clf.parameters())
        # maybe unnecessary
        set_grads_to_none(z_alpha_X)
        set_grads_to_none(gen.parameters())

        dists_per_example = compute_dist(
            dist_fn,
            val_fn,
            this_acts,
            orig_acts_grads,
        )
        dists = th.mean(th.stack(dists_per_example), dim=1)
        p_counts = p_counts = [1] * len(dists)
        assert len(p_counts) == len(dists)
        task_loss = weighted_sum(
            len(dists), *list(itertools.chain(*list(zip(p_counts, dists))))
        )
        bpd = get_bpd(gen, aug_m(simple_X))
        bpd_loss = th.mean(bpd)

        mse_z = sum(
            [
                th.sum(th.square(z_orig - z_simple))
                for z_orig, z_simple in zip(z_orig_X, z_alpha_X)
            ]
        ) / np.prod(X.shape)

        loss = weighted_sum(
            1,
            bpd_weight,
            bpd_loss,
            task_weight,
            task_loss,
            mse_z_weight,
            mse_z,
            1,
            out_of_0_1_loss,
        )

        opt_simple_X_alpha.zero_grad(set_to_none=True)
        loss.backward()
        finite_grads = grads_all_finite(opt_simple_X_alpha)
        if finite_grads:
            opt_simple_X_alpha.step()
        opt_simple_X_alpha.zero_grad(set_to_none=True)

        with th.no_grad():
            simple_out = this_acts[wanted_modules[-1]]["out_act"][0]  # clf(simple_X)
            # could also add this to loss actually
            kl_div_orig_simple = kl_divergence(orig_out, simple_out)
        weighted_task_loss = task_weight * task_loss
        if finite_grads:
            results = dict(
                loss=loss.item(),
                bpd_loss=bpd_loss.item(),
                task_loss=task_loss.item(),
                weighted_task_loss=weighted_task_loss.item(),
                mean_dist=th.mean(dists).item(),
                mean_weighted_dist=weighted_sum(
                    1, *list(itertools.chain(*list(zip(p_counts, dists))))
                ).item(),
                kl_div_orig_simple=kl_div_orig_simple.item(),
                mse_z=mse_z.item(),
                out_of_0_1_loss=out_of_0_1_loss.item(),
            )
            nb_res.collect(**results)
            nb_res.print()
    this_acts = detach_acts_grads(this_acts)


    ## Old style image, original images on top of simplified
    with th.no_grad():
        simple_X = glow.invert(z_alpha_X)[0] + 0.5
        simple_out = clf(simple_X)
        soft_orig_out = th.softmax(orig_out, dim=1)
        soft_simple_out = th.softmax(simple_out, dim=1)
        simple_pred_y = th.argmax(simple_out, dim=1)
    im_arr = stack_images_in_rows(
        th_to_np(this_X), th_to_np(simple_X), n_cols=min(len(this_X), 8)
    )
    padded_im_arr = np.pad(
        im_arr,
        ((0, 0), (0, 0), (0, 0), (24, 0), (0, 0)),
        constant_values=0.8,
    )
    im = create_rgb_image(padded_im_arr)


    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

    labels = [valid_dataset.classes[a_y][0] for a_y in this_y]
    pred_labels = [valid_dataset.classes[i][0] for i in simple_out.argmax(dim=1)]
    wanted_labels = [valid_dataset.classes[i][0] for i in wanted_y]

    for i_example, (label, pred_label) in enumerate(zip(labels, pred_labels)):
        i_col = i_example % 8
        i_row = i_example // 8

        draw.text(
            (
                i_col * 224,
                i_row * (224 + 24) * 2 + 2,
            ),
            f'{label.replace("_", " ").capitalize()} {soft_orig_out[i_example, this_y[i_example]].item():.1%}',
            (0, 0, 0),
            font=font,
        )
        pred_str = f'{pred_label.replace("_", " ").capitalize()} {soft_simple_out[i_example, simple_pred_y[i_example]].item():.1%}'
        if simple_pred_y[i_example] != wanted_y[i_example]:
            pred_str += f' ({wanted_labels[i_example].replace("_", " ").capitalize()} {soft_simple_out[i_example, wanted_y[i_example]].item():.1%})'
        draw.text(
            (
                i_col * 224,
                i_row * (224 + 24) * 2 + (224 + 24) + 2,
            ),
            pred_str,
            (0, 0, 0),
            font=font,
        )
    im.save(os.path.join(output_dir, "pred_image.png"))

    th.save(simple_X.detach(), os.path.join(output_dir, "simple_X.th"))
    th.save(orig_out.detach(), os.path.join(output_dir, "orig_out.th"))
    th.save(simple_out.detach(), os.path.join(output_dir, "simple_out.th"))
    th.save(wanted_y.detach(), os.path.join(output_dir, "wanted_y.th"))
    th.save(this_X.detach(), os.path.join(output_dir, "this_X.th"))
    th.save(this_y.detach(), os.path.join(output_dir, "this_y.th"))

    ## New style image, original image next to simplified

    with th.no_grad():
        
        orig_out = clf(this_X)
        soft_orig_out = th.softmax(orig_out, dim=1)
        pred_y = soft_orig_out.argmax(dim=1)
        simple_X = glow.invert(z_alpha_X)[0] + 0.5
        simple_out = clf(simple_X)
        soft_simple_out = th.softmax(simple_out, dim=1)
        simple_pred_y = th.argmax(simple_out, dim=1)

    im_arrs = []
    for i_example in range(0, len(this_X), 2):
        assert th.allclose(this_X[i_example], this_X[i_example+1],rtol=1e-2,atol=1e-3)
        im_arrs.append([th_to_np(this_X[i_example]),
                        th_to_np(simple_X[i_example]),
                        th_to_np(simple_X[i_example+1])])

    im_arr = np.stack(im_arrs, axis=0)

    padded_im_arr = np.pad(
        im_arr,
        ((0, 0), (0, 0), (0, 0), (48, 0), (0, 0)),
        constant_values=0.8,
    )
    im = create_rgb_image(padded_im_arr)


    labels = [valid_dataset.classes[a_y][0] for a_y in this_y]
    orig_pred_labels = [valid_dataset.classes[i][0] for i in pred_y]
    simple_pred_labels = [valid_dataset.classes[i][0] for i in simple_pred_y]
    wanted_labels = [valid_dataset.classes[i][0] for i in wanted_y]

    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12
    )

    for i_example in range(0, len(this_X), 2):
        i_row = i_example // 2
        assert th.allclose(this_X[i_example], this_X[i_example+1],rtol=1e-2,atol=1e-3)
        i_col = 0
        draw.text(
            (
                i_col * 224,
                i_row * (224 + 48) + 2,
            ),
            
            f'True: {labels[i_example].replace("_", " ").capitalize():18s} {soft_orig_out[i_example, this_y[i_example]].item():5.1%}\n'+
            f'Pred: {orig_pred_labels[i_example].replace("_", " ").capitalize():18s} {soft_orig_out[i_example, pred_y[i_example]].item():5.1%}',
            (0, 0, 0),
            font=font,
        )
        for i_simple in (i_example,i_example+1):
            i_col = (i_simple % 2) + 1
            pred_str = f'{simple_pred_labels[i_simple].replace("_", " ").capitalize():18s} {soft_simple_out[i_simple, simple_pred_y[i_simple]].item():5.1%}'
            if simple_pred_y[i_simple] != wanted_y[i_simple]:
                pred_str += f'\n{wanted_labels[i_simple].replace("_", " ").capitalize():18s} {soft_simple_out[i_simple, wanted_y[i_simple]].item():5.1%}'
            draw.text(
                (
                    i_col * 224,
                    i_row * (224 + 48) + 2,
                ),
                pred_str,
                (0, 0, 0),
                font=font,
            )
    im.save(os.path.join(output_dir, "pred_image_single_rows.png"))

    results = dict(
        loss=loss.item(),
        bpd_loss=bpd_loss.item(),
        task_loss=task_loss.item(),
        weighted_task_loss=weighted_task_loss.item(),
        mean_dist=th.mean(dists).item(),
        mean_weighted_dist=weighted_sum(
            1, *list(itertools.chain(*list(zip(p_counts, dists))))
        ).item(),
        kl_div_orig_simple=kl_div_orig_simple.item(),
        mse_z=mse_z.item(),
        out_of_0_1_loss=out_of_0_1_loss.item(),
        acc=acc.item(),
    )
    return results