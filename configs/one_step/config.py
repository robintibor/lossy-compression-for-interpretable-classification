import os

os.sys.path.insert(0, "/home/schirrmr/code/utils/")
os.sys.path.insert(0, "/home/schirrmr/code/lossy/")
os.sys.path.insert(0, "/home/schirrmr/code/nfnets/")
# os.sys.path.insert(0, "/home/schirrmr/code/invertible-neurips/")
os.sys.path.insert(0, "/home/schirrmr/code/cifar10-clf/")
import time
import logging

from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)

import torch as th
import torch.backends.cudnn as cudnn

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s")


log = logging.getLogger(__name__)
log.setLevel("INFO")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/",  # before rebuttal without "icml-"
            # "/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    # data_params = [
    #     {
    #         'dataset': 'mnist',
    #         'saved_model_folder': '/home/schirrmr/data/exps/lossy/mnist-wide-nfnets/20/'
    #     },
    #     {
    #         'dataset': 'fashionmnist',
    #         'saved_model_folder': '/home/schirrmr/data/exps/lossy/mnist-wide-nfnets/21/'
    #
    #     },
    #     {
    #         'dataset': 'cifar10',
    #         #'saved_model_folder': '/home/schirrmr/data/exps/lossy/cifar10-wide-nfnets/114/'
    #          'saved_model_folder': '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/cifar10-wide-nfnets-shifted-softplus/23/'
    #     }
    # ]

    data_params = dictlistprod(
        {
            "dataset": ["cifar10", "mnist", "fashionmnist", "svhn"],  # , 'mnist', 'fashionmnist', 'svhn'],
            #'dataset': ['cifar10'],#, 'mnist'],
            "saved_model_folder": [
                None
            ],  # ['/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/cifar10-wide-nfnets-shifted-softplus/23/'],#[None],
            #'saved_model_folder': [None],#['/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/simple-convnets/2/'],  # [None],
            "mimic_cxr_target": [None],
            "first_n": [None],
        }
    )

    train_params = dictlistprod(
        {
            # "n_epochs": [2],
            "n_epochs": [100],
            "batch_size": [32],
            "train_orig": [False],
            "train_simclr_orig": [False],
            "train_ssl_orig_simple": [False],
            "ssl_loss_factor": [None],
            "grad_from_orig": [True],  # True
            "use_normed_loss": [False],  # False
            "use_expected_loss": [False,],  # False
            "separate_orig_clf": [True],
            "simple_orig_pred_loss_weight": [0],  # 4
            "scale_dists_loss_by_n_vals": [False],
            "dist_name": ["normed_sse"],
            "conv_grad_name": ["loop"],  # loop backpack
        }
    )

    loss_and_dist_params = [
        {
            "loss_name": "grad_act_match",  # , ""],#'gradparam_param',
            "per_module": True,
            "per_model": False,
        },
        # {
        #     "per_module": False,
        #     "per_model": True,
        #     "loss_name": "gradparam_param",
        # },
    ]

    # dist_params = [
    # #     {
    # #     "per_module": False,
    # #     "per_model": False,
    # # },
    #     {
    #     "per_module": True,
    #     "per_model": False,
    # },
    # #     {
    # #     "per_module": False,
    # #     "per_model": True,
    # # },
    # ]

    noise_params = [
        {
            "noise_augment_level": 0,
            "trivial_augment": False,
            "extra_augs": False,
        },
    ]

    quantize_params = [
        #     {
        #     "noise_after_simplifier": False,
        #     "noise_before_generator": True,
        #     "np_th_seed": 0,
        #     'quantize_after_simplifier': True,
        # },
        {
            "noise_after_simplifier": True,
            "noise_before_generator": False,
            "np_th_seed": 0,
            "quantize_after_simplifier": True,
        }
    ]

    # quantize_params = dictlistprod(
    #     {
    #         "np_th_seed": range(1),
    #         #"np_th_seed": [0],
    #         'quantize_after_simplifier': [True,False],
    #     }
    # ) + dictlistprod(
    #     {
    #         "np_th_seed": range(1,3),
    #         #"np_th_seed": [0],
    #         'quantize_after_simplifier': [True,],
    #     }
    # )

    random_params = dictlistprod(
        {
            # "np_th_seed": [0],
        }
    )

    # model_params = dictlistprod(
    #     {
    #         "depth": [16],
    #         "widen_factor": [2],
    #         "n_start_filters": [64],
    #         "residual_preproc": [
    #             True,
    #         ],
    #         "model_name": ["wide_nf_net"],
    #         "adjust_betas": [False],
    #         'save_models': [True],
    #         'activation': ["shifted_softplus_1"],
    #         "norm_simple_convnet": ['none'],
    #         "pooling": ['avgpooling'],
    #     }
    # )

    model_params = dictlistprod(
        {
            "depth": [16],
            "widen_factor": [2],
            "n_start_filters": [64],
            "residual_preproc": [
                True,
            ],
            "model_name": ["wide_nf_net"],
            "adjust_betas": [False],
            "save_models": [True],
            "activation": ["shifted_softplus_1"],
            "norm_simple_convnet": ["none"],
            "pooling": ["avgpooling"],
        }
    )

    optim_params = dictlistprod(
        {
            "resample_augmentation": [False],  # default this was True
            "resample_augmentation_for_clf": [False],  # default this was False
            "std_aug_magnitude": [None],  # 0.25
            "weight_decay": [1e-5],
            "lr_clf": [
                1e-3,
            ],  # 5e-4,
            "lr_preproc": [
                5e-4,
            ],  # 5e-4,
            "threshold": [
                0.1,
            ],
            "optim_type": [
                "adamw",
            ],
            "bpd_weight": [
                0.7,
                0.8,
                0.9,
                1.0
            ],  # [0.32,0.34,0.36,0.38,0.4],#[0.3,0.333,0.367,0.4,0.433,0.467,0.5],#[0., 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0],
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            train_params,
            random_params,
            loss_and_dist_params,
            debug_params,
            model_params,
            optim_params,
            data_params,
            quantize_params,
            noise_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    n_epochs,
    optim_type,
    n_start_filters,
    residual_preproc,
    lr_preproc,
    lr_clf,
    threshold,
    bpd_weight,
    model_name,
    np_th_seed,
    batch_size,
    weight_decay,
    adjust_betas,
    saved_model_folder,
    train_orig,
    dataset,
    debug,
    save_models,
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
    first_n,
    scale_dists_loss_by_n_vals,
    per_module,
    per_model,
    norm_simple_convnet,
    pooling,
    dist_name,
    conv_grad_name,
    use_expected_loss,
):
    if debug:
        n_epochs = 3
        first_n = 1024
        save_models = False
    kwargs = locals()
    kwargs.pop("ex")
    kwargs.pop("debug")
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
    os.environ['imagenet'] = "/data/datasets/ImageNet/imagenet-pytorch/"
    from lossy.experiments.one_step.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
