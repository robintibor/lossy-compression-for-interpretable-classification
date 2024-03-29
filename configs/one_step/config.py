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

    data_params = dictlistprod(
        {  
            "mimic_cxr_target": ['pleural_effusion'],  # ]'pleural_effusion'],
            "first_n": [None],
            "stripes_factor": [0.3],
        }
    )

    data_model_params = [
        # {
        #     "dataset": "cifar10",  # , 'mnist', 'fashionmnist', 'svhn'],
        #     "saved_model_folder": '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/cifar10-wide-nfnets-shifted-softplus/23/',
        #     #['/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/simple-convnets/2/'],  # [None],
        # },
        # {
        #     "dataset": "svhn",  # , 'mnist', 'fashionmnist', 'svhn'],
        #     "saved_model_folder": '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/nf-net-stripes/25/',
        # },
        # {
        #     "dataset": "mnist",  # , 'mnist', 'fashionmnist', 'svhn'],
        #     "saved_model_folder": '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/nf-net-stripes/26/',
        # },
        # {
        #     "dataset": "fashionmnist",  # , '', 'fashionmnist', 'svhn'],
        #     "saved_model_folder": '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/nf-net-stripes/27/',
        # },


    ]

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
            "clf_loss_name": ["expected_grad_loss"],  # False
            "simple_orig_pred_loss_weight": [0],  # 4
            "scale_dists_loss_by_n_vals": [False],
            "conv_grad_name": ["loop"],  # loop backpack
            "dist_threshold": [0.05, 0.1, 0.2, 0.3, 0.4,0.5],  # ],#0.05,
            "dist_margin":  [1e-5],#0.1
            "pretrain_clf_epochs": [0],
            "detach_bpd_factors": [True],
            "frozen_clf": [False],
            "first_batch_only": [False],
            "simple_clf_loss_threshold": [None],
            "threshold_simple_class_correct": [True],
            "bound_grad_norm_factor": [None],
            "skip_unneeded_bpd_computations": [True],
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

    noise_params = dictlistprod(
        {
            "noise_augment_level": [0],  # 0,
            "trivial_augment": [False],
            "extra_augs": [False],
        },
    )

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

    model_params = dictlistprod(
        {
            "depth": [16],
            "widen_factor": [2],
            "n_start_filters": [64],
            "model_name": ["vit"],#["wide_nf_net"],
            "adjust_betas": [False],
            "save_models": [True],
            "activation": ["shifted_softplus_1"],
            "norm_simple_convnet": ["none"],
            "pooling": ["avgpooling"],
            "external_pretrained_clf": [False],
            "glow_model_path_32x32": [
                "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
            ],  # /home/schirrmr/data/exps/invertible-neurips/smaller-glow/22/10_model.th
            "soft_clamp_0_1": [True],
            "unet_use_bias": [True],
            #"preproc_glow_path": [None, "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/22/10_model.th"],
        }
    )

    preproc_params = (
        dictlistprod(
            {
                # "lr_preproc": [1e-4,],
                # "preproc_name": ["glow_with_resnet"],#res_unet
                # "cat_clf_chans_for_preproc": [False],
                # "merge_weight_clf_chans": [None],
                # "n_pretrain_preproc_epochs": [0],
            }
        )
        + dictlistprod(
            {
                # "lr_preproc": [3e-4,],
                # "preproc_name": ["unet",],#res_unet
                # "cat_clf_chans_for_preproc": [False],
                # "merge_weight_clf_chans" [None],
                # "n_pretrain_preproc_epochs": [0],
            }
        )
        + dictlistprod(
            {
                # "lr_preproc": [5e-4,],
                # "preproc_name": ["res_unet",],#res_unet
                # "cat_clf_chans_for_preproc": [False],
                # "merge_weight_clf_chans": [None],
                # "n_pretrain_preproc_epochs": [0],
            }
        )
        + dictlistprod(
            {
                # "lr_preproc": [3e-4,],
                # "preproc_name": ["on_top_of_glow",],#res_unet
                # "cat_clf_chans_for_preproc": [False],
                # "merge_weight_clf_chans": [None],
                # "n_pretrain_preproc_epochs": [0],
            }
        )
        + dictlistprod(
            {
                # "lr_preproc": [3e-4,],
                # "preproc_name": ["res_blend_unet",],#res_unet
                # "cat_clf_chans_for_preproc": [False],
                # "merge_weight_clf_chans": [None],
                # "n_pretrain_preproc_epochs": [0],
            }
        )
        + dictlistprod(
            {
               # "lr_preproc": [
               #     3e-4,
               # ],
               # "preproc_name": [
               #     "res_mix_unet",
               # ],  # res_unet
               # "cat_clf_chans_for_preproc": [False],
               # "merge_weight_clf_chans": [None],
               # "n_pretrain_preproc_epochs": [0],
            }
        )
        + dictlistprod(
            {
                # "lr_preproc": [1e-4,],
                # "preproc_name": ["glow_with_pure_resnet"],#res_unet
                # "cat_clf_chans_for_preproc": [False],#, True],
                # "merge_weight_clf_chans": [1e-2],
                # "n_pretrain_preproc_epochs": [2,],
                # "encoder_clip_eps": [1e-1],
            }
        )
        + dictlistprod(
            {
                "lr_preproc": [1e-4,],
                "preproc_name": ["res_glow_with_pure_resnet"],#res_unet
                "cat_clf_chans_for_preproc": [False],#, True],
                "merge_weight_clf_chans": [1e-2],
                "n_pretrain_preproc_epochs": [0,],
                "encoder_clip_eps": [1e-1],
            }
        )
        + dictlistprod(
            {
                # "lr_preproc": [3e-4,],
                # "preproc_name": ["res_mix_grey_unet",],#res_unet
                # "cat_clf_chans_for_preproc": [False],
                # "merge_weight_clf_chans": [None],
                # "n_pretrain_preproc_epochs": [0],
            }
        )
    )

    optim_params = dictlistprod(
        {
            "resample_augmentation": [False],  # default this was True
            "resample_augmentation_for_clf": [False],  # default this was False
            "std_aug_magnitude": [None],  # 0.25
            "weight_decay": [1e-5],  # [1e-5],
            "clip_grad_percentile": [80],
            "lr_clf": [
                1e-3,
            ],  # 5e-4,
            "weight_decay_preproc": [5e-5],
            "threshold": [
                0.1,
            ],
            "optim_type": [
                "adamw",
            ],
            "bpd_weight": [
                # 0.7,
                # 0.8,
                # 0.9,
                4,
                # 0.5,
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
            data_model_params,
            quantize_params,
            noise_params,
            preproc_params,
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
    external_pretrained_clf,
    clf_loss_name,
    orig_loss_weight,
    pretrain_clf_epochs,
    preproc_name,
    train_clf_on_dist_loss,
    train_clf_on_orig_simultaneously,
    dist_threshold,
    glow_model_path_32x32,
    detach_bpd_factors,
    stop_clf_grad_through_simple,
    simple_clf_loss_weight,
    soft_clamp_0_1,
    unet_use_bias,
    frozen_clf,
    first_batch_only,
    cat_clf_chans_for_preproc,
    merge_weight_clf_chans,
    weight_decay_preproc,
    n_pretrain_preproc_epochs,
    encoder_clip_eps,
    clip_grad_percentile,
    dist_margin,
    stripes_factor,
    simple_clf_loss_threshold,
    threshold_simple_class_correct,
    bound_grad_norm_factor,
    skip_unneeded_bpd_computations,
    preproc_glow_path,
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
    os.environ["imagenet"] = "/data/datasets/ImageNet/imagenet-pytorch/"
    from lossy.experiments.one_step.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
