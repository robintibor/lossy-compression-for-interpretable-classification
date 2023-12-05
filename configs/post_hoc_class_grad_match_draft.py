import os

os.sys.path.insert(0, "/home/schirrmr/code/utils/")
os.sys.path.insert(0, "/home/schirrmr/code/lossy/")
os.sys.path.insert(0, "/home/schirrmr/code/invertible-neurips/")
import time
import logging

from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)
from hyperoptim.results import load_data_frame
import torch as th
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import transforms

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s")


log = logging.getLogger(__name__)
log.setLevel("INFO")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/post-hoc-draft-diff-models/",
            # "/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    root = "/data/datasets/ImageNet/imagenet-pytorch"

    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


    os.environ["pytorch_data"] = "/home/schirrmr/data/pytorch-datasets/"
    os.environ[
        "mimic_cxr"
    ] = "/work/dlclarge2/schirrmr-mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    os.environ[
        "small_glow_path"
    ] = "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
    os.environ[
        "normal_glow_path"
    ] = "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
    os.environ["imagenet"] = "/data/datasets/ImageNet/imagenet-pytorch/"
    from lossy.datasets import ImageNet
    # split = "train"
    # dataset = ImageNet(
    #     root=root,
    #     split=split,
    #     transform=valid_transform,
    #     ignore_archive=True,
    # )

    # class_a = 46
    # class_b = 40
    # ys_from_imgs = np.array([img[1] for img in dataset.imgs])
    # inds = np.flatnonzero((ys_from_imgs == class_a) | (ys_from_imgs == class_b))
    split = "val"
   
    # inds = np.arange(0,50000,500)
    inds = [2383, 2968, 1356, 1075, 10, 293, 355, 367, 506]

    data_params = dictlistprod(
        {
            "image_ind": inds,
            # "image_ind": np.concatenate(
            #     [
            #         np.arange(100 + i_start, 1281167, 1281167 // 100)
            #         for i_start in range(11, 12)
            #     ]
            # ),
            # np.arange(100,1281167, 1281167 // 100),#range(100),

            "split": [split],
        }
    )
    # 273,1057

    model_params = dictlistprod(
        {
            "model_name": [
                "resnet18",  "resnet50", "resnet101",  "wide_resnet50_2",
            ],
            "softplus_beta": [4],#[4],
        }
    )

    train_params = dictlistprod(
        {
            "n_epochs": [700],
            "dist_threshold": [1],#0.4],
            "wanted_y": [None],#[None, None]],
            "n_top_classes": [2],
            "add_true_class": [True],
            "bpd_weight": [3e-2],
            "orig_factor": [1],
            "start_pixel_val": [0.9],
        }
    )

    random_params = dictlistprod(
        {
            "np_th_seed": range(1),
        }
    )
    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            train_params,
            data_params,
            random_params,
            model_params,
            debug_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    model_name,
    n_epochs,
    image_ind,
    dist_threshold,
    np_th_seed,
    debug,
    softplus_beta,
    n_top_classes,
    split,
    wanted_y,
    add_true_class,
    start_pixel_val,
    orig_factor,
    bpd_weight,
):
    if debug:
        n_epochs = 3
    kwargs = locals()
    kwargs.pop("ex")
    if not debug:
        log.setLevel("INFO")
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    kwargs["output_dir"] = output_dir
    th.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    start_time = time.time()
    ex.info["finished"] = False

    import os

    os.environ["pytorch_data"] = "/home/schirrmr/data/pytorch-datasets/"
    os.environ[
        "mimic_cxr"
    ] = "/work/dlclarge2/schirrmr-mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    os.environ[
        "small_glow_path"
    ] = "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
    os.environ[
        "normal_glow_path"
    ] = "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
    os.environ["imagenet"] = "/data/datasets/ImageNet/imagenet-pytorch/"

    from lossy.experiments.post_hoc_class_grad_match_draft import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
