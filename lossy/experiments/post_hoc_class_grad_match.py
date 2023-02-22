import logging
import logging
import os.path
from copy import deepcopy

import numpy as np
import torch
import torch as th
import torchvision.models as models
from PIL import ImageDraw, ImageFont
from rtsutils.nb_util import Results
from rtsutils.optim import grads_all_finite
from rtsutils.plot import create_rgb_image
from rtsutils.util import th_to_np
from torch import nn
from torchvision import transforms

from lossy.activation_match import compute_dist
from lossy.activation_match import cosine_distance
from lossy.activation_match import detach_acts_grads
from lossy.activation_match import get_in_out_activations_per_module
from lossy.activation_match import get_in_out_acts_and_in_out_grads_per_module
from lossy.activation_match import grad_in_act_act
from lossy.activation_match import relu_match, refed
from lossy.losses import expected_grad_loss
from lossy.losses import kl_divergence
from lossy.optim import grads_all_finite
from lossy.softplus import ReLUSoftPlusGrad
from lossy.util import set_random_seeds
from lossy.util import weighted_sum

log = logging.getLogger(__name__)



def run_exp(
    model_name,
    n_epochs,
    image_ind,
    np_th_seed,
    dist_threshold,
    debug,
    output_dir,
    softplus_beta,
    n_top_classes,
    split,
    wanted_y
):
    assert (wanted_y is None) != (n_top_classes is None)
    trange = range
    set_random_seeds(np_th_seed, True)
    from lossy.datasets import ImageNet

    # https://github.com/pytorch/examples/blob/fcf8f9498e40863405fe367b9521269e03d7f521/imagenet/main.py#L213-L237

    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    log.info("Loading data...")

    root = "/data/datasets/ImageNet/imagenet-pytorch"

    dataset = ImageNet(
        root=root,
        split=split,
        transform=valid_transform,
        ignore_archive=True,
    )

    n_x = len(wanted_y) if n_top_classes is None else n_top_classes
    orig_X, orig_y = dataset[image_ind]

    this_X = th.cat((orig_X.unsqueeze(0),) * n_x).cuda()
    this_X = this_X.clone().requires_grad_(True)

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

    with th.no_grad():
        orig_preds = orig_clf(this_X[0:1])
    if wanted_y is None:
        wanted_y = orig_preds.squeeze().argsort(descending=True)[:n_top_classes]
    else:
        wanted_y = th.tensor(wanted_y).cuda()


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

    with torch.no_grad():
        z_orig_X = gen(this_X)[0]
        z_grey = gen(torch.zeros_like(this_X) + 0.5)[0]

    orig_factor = 0.25

    z_alpha_X = [
        (a_z_grey * (1 - orig_factor) + z_orig * orig_factor)
        .clone()
        .requires_grad_(True)
        for a_z_grey, z_orig in zip(z_grey, z_orig_X)
    ]

    opt_simple_X_alpha = torch.optim.Adam(
        [
            dict(params=z_alpha_X, lr=1e-2),
        ]
    )

    loss_fn = lambda o: -expected_grad_loss(o, wanted_y)

    orig_acts_grads = get_in_out_acts_and_in_out_grads_per_module(
        clf, this_X, loss_fn, wanted_modules=wanted_modules
    )

    orig_acts_grads = detach_acts_grads(orig_acts_grads)
    with torch.no_grad():
        orig_out = clf(this_X)

    bpd_weight = 1e-1

    # val_fn = relu_match(refed(grad_in_act_act, 'in_grad'))
    val_fn = relu_match(refed(grad_in_act_act, "in_grad"))
    dist_fn = cosine_distance  # normed_sse#cosine_distance#larger_magnitude_cos_dist#cos_dist_fn#normed_mse#mse_dist_fn#l1_loss#sse_dist_fn#
    use_parameter_counts = True

    def aug_m(x):
        return x * 255 / 256.0 + torch.rand_like(x) * (1 / 256.0)

    nb_res = Results(0.95)
    for i_epoch in trange(n_epochs):
        glow.remove_cur_in_out()
        clf.eval()
        simple_X = glow.invert(z_alpha_X)[0] + 0.5

        out_of_0_1_loss = (
            torch.nn.functional.relu(torch.abs(simple_X - 0.5) - 0.5).square().sum()
        )
        simple_X = torch.clamp(simple_X, 0, 1)
        # Get only forward acts and then copy into appropriate structure
        this_acts = get_in_out_activations_per_module(
            clf,
            aug_m(simple_X),
            wanted_modules=wanted_modules,
        )

        dists = compute_dist(
            dist_fn,
            val_fn,
            this_acts,
            orig_acts_grads,
        )

        dists = torch.stack(dists, dim=1)  # examples x layers
        if use_parameter_counts:
            p_counts = [
                sum([p.numel() for p in m.parameters(recurse=False)])
                for m in wanted_modules
            ]
        else:
            p_counts = [1] * dists.shape[1]
        assert len(p_counts) == dists.shape[1]
        normalized_p_counts = torch.tensor(p_counts) / torch.tensor(p_counts).sum()
        normalized_p_counts = normalized_p_counts.to(dists.device)
        weighted_dists = torch.einsum("l,il->il", normalized_p_counts, dists)

        dists_per_example = torch.sum(weighted_dists, dim=1)
        dists_per_layer = torch.mean(dists, dim=0)  # here do not take layerweighting
        task_loss = torch.mean(dists_per_example)
        bpd_factors = 1.0 * (dists_per_example < dist_threshold).detach()

        bpd = get_bpd(gen, aug_m(simple_X))
        weighted_bpd_loss = torch.mean(bpd * bpd_factors)

        loss = weighted_sum(
            1, bpd_weight, weighted_bpd_loss, 1, task_loss, bpd_weight, out_of_0_1_loss
        )

        opt_simple_X_alpha.zero_grad(set_to_none=True)
        loss.backward()
        finite_grads = grads_all_finite(opt_simple_X_alpha)
        if finite_grads:
            opt_simple_X_alpha.step()
        opt_simple_X_alpha.zero_grad(set_to_none=True)

        with torch.no_grad():
            bpd_loss = torch.mean(bpd)
        with torch.no_grad():
            simple_out = this_acts[wanted_modules[-1]]["out_act"][0]  # clf(simple_X)
            # could also add this to loss actually
            kl_div_orig_simple = kl_divergence(orig_out, simple_out)
        if finite_grads:
            results = dict(
                loss=loss.item(),
                weighted_bpd_loss=weighted_bpd_loss.item(),
                bpd_loss=bpd_loss.item(),
                task_loss=task_loss.item(),
                mean_dist=torch.mean(dists).item(),
                mean_weighted_dist=torch.mean(dists_per_example).item(),
                kl_div_orig_simple=kl_div_orig_simple.item(),
                out_of_0_1_loss=out_of_0_1_loss.item(),
            )
            nb_res.collect(**results)
            # nb_res.print()
            if i_epoch % max(1, n_epochs // 10) == 0:
                print(f"Epoch {i_epoch}")

    with torch.no_grad():
        simple_X = glow.invert(z_alpha_X)[0] + 0.5
        simple_out = clf(simple_X)
        soft_orig_out = torch.softmax(orig_out, dim=1)
        soft_simple_out = torch.softmax(simple_out, dim=1)
        simple_pred_y = torch.argmax(simple_out, dim=1)

    th.save(simple_X.detach(), os.path.join(output_dir, "simple_X.th"))
    th.save(orig_out.detach(), os.path.join(output_dir, "orig_out.th"))
    th.save(simple_out.detach(), os.path.join(output_dir, "simple_out.th"))
    th.save(wanted_y.detach(), os.path.join(output_dir, "wanted_y.th"))
    th.save(this_X.detach(), os.path.join(output_dir, "this_X.th"))

    ## New style image, original image next to simplified

    with th.no_grad():

        orig_out = clf(this_X)
        soft_orig_out = th.softmax(orig_out, dim=1)
        pred_y = soft_orig_out.argmax(dim=1)
        simple_X = glow.invert(z_alpha_X)[0] + 0.5
        simple_out = clf(simple_X)
        soft_simple_out = th.softmax(simple_out, dim=1)
        simple_pred_y = th.argmax(simple_out, dim=1)

    # Check that this_X is all same actually
    assert th.allclose(
        th.zeros_like(this_X[:-1]), th.diff(this_X, dim=0), rtol=1e-5, atol=1e-5
    )
    im_arr = np.concatenate((th_to_np(this_X[0:1]), th_to_np(simple_X)))[None]

    padded_im_arr = np.pad(
        im_arr,
        ((0, 0), (0, 0), (0, 0), (48, 0), (0, 0)),
        constant_values=0.8,
    )
    im = create_rgb_image(padded_im_arr)

    # hacky for now
    labels = [dataset.classes[a_y][0] for a_y in (orig_y,) * len(wanted_y)]
    orig_pred_labels = [dataset.classes[i][0] for i in pred_y]
    simple_pred_labels = [dataset.classes[i][0] for i in simple_pred_y]
    wanted_labels = [dataset.classes[i][0] for i in wanted_y]

    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)

    i_row = 0
    i_col = 0
    draw.text(
        (
            i_col * 224,
            i_row * (224 + 48) + 2,
        ),
        f'True: {labels[0].replace("_", " ").capitalize():18s} {soft_orig_out[0, orig_y].item():5.1%}\n'
        + f'Pred: {orig_pred_labels[0].replace("_", " ").capitalize():18s} {soft_orig_out[0, pred_y[0]].item():5.1%}',
        (0, 0, 0),
        font=font,
    )
    for i_simple in range(len(this_X)):
        i_col = i_simple + 1
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
        mean_dist=th.mean(dists).item(),
        mean_weighted_dist=torch.mean(dists_per_example).item(),
        kl_div_orig_simple=kl_div_orig_simple.item(),
        out_of_0_1_loss=out_of_0_1_loss.item(),
    )

    return results
