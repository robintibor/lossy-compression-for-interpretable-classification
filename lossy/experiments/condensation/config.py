import torch.backends.cudnn as cudnn
from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)
import logging
import time
import os

os.sys.path.insert(0, "/home/schirrmr/code/lossy/")
os.sys.path.insert(0, "/home/schirrmr/code/invertible-neurips/")
os.sys.path.insert(0, '/home/schirrmr/code/cifar10-clf/')

import numpy as np
import torch
from braindecode.util import set_random_seeds


logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s")


log = logging.getLogger(__name__)
log.setLevel("INFO")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/condensation/mimic-cxr-25000-train-data/",
        },
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    data_params = dictlistprod({
        'dataset': ['MIMIC-CXR'],#, 'MNIST', 'SVHN', 'FashionMNIST'
        'mimic_cxr_clip': [0.6,1.0],
    })

    ipc_params = [
        {
            "ipc": 1,
            "outer_loop": 1,
            "inner_loop": 1,
        },
        {
            "ipc": 1,
            "outer_loop": 5,
            "inner_loop": 5,
        },
        # {
        #     "ipc": 10,
        #     "outer_loop": 10,
        #     "inner_loop": 50,
        # }
    ]

    train_params = dictlistprod(
        {
            "glow_noise_on_out": [True],
            "n_outer_epochs": [1000],
            "bpd_loss_weight": [0, 10, 100, 1000,10000],#[100000,1000000],
            "img_alpha_init_factor": [
                0.2,
            ],  # 10.1,0.4
            "image_standardize_before_glow": [
                False,
            ],  # True
            "sigmoid_on_alpha": [True],  # False#FalseFalse
            "lr_img": [
                0.1,
            ],  # 1,0.05,
            "lr_net": [
                1e-2,
            ],  # 0.05
            "standardize_for_clf": [False],  # TrueTrue
            "net_norm": ["instancenorm"],#["no
            # ne", "instancenorm"],
            "net_act": ["relu"],#"relu",
            "optim_class_img": ["sgd", "adam"],#"sgd",
            "loss_name": ["match_loss"],#"grad_grad",
            "rescale_grads": [False,],
            "saved_model_path": [None],#"chenyaofo/pytorch-cifar-models#cifar100_resnet20"
            "trivial_augment": [False],
            "same_aug_across_batch": [False],
        }
    )

    random_params = dictlistprod(
        {
            "seed": [0],#,1,2],  # range(0, 3),
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            data_params,
            ipc_params,
            train_params,
            debug_params,
            random_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    n_outer_epochs,
    bpd_loss_weight,
    ipc,
    img_alpha_init_factor,
    image_standardize_before_glow,
    seed,
    debug,
    sigmoid_on_alpha,
    lr_img,
    lr_net,
    standardize_for_clf,
    net_norm,
    net_act,
    loss_name,
    optim_class_img,
    outer_loop,
    inner_loop,
    saved_model_path,
    rescale_grads,
    dataset,
    glow_noise_on_out,
    trivial_augment,
    same_aug_across_batch,
    mimic_cxr_clip,
):
    data_path = '/home/schirrmr/data/pytorch-datasets/'
    model_name = "ConvNet"
    batch_real = 256
    batch_train = 256
    init = "noise"
    dis_metric = "ours"
    epoch_eval_train = 300
    num_exp = 1
    num_eval = 5
    eval_mode = "S"
    kwargs = locals()
    kwargs.pop("ex")
    if not debug:
        log.setLevel("INFO")
    if debug:
        kwargs["n_outer_epochs"] = 3
    kwargs.pop("debug")
    set_random_seeds(seed, True)
    kwargs.pop("seed")

    file_obs = ex.observers[0]
    save_path = file_obs.dir
    kwargs["save_path"] = save_path
    torch.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )

    start_time = time.time()
    ex.info["finished"] = False

    from lossy.experiments.condensation.run import run_exp

    results = run_exp(**kwargs)

    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    ex.info["acc_mean"] = float(np.mean(results["accs"]["ConvNet"]))
    ex.info["acc_std"] = float(np.std(results["accs"]["ConvNet"]))
    ex.info["bpd"] = float(results["bpd"])
    ex.info["runtime"] = run_time
