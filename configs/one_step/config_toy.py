# PARENTCONFIG: /home/schirrmr/code/lossy/configs/one_step/config.py

import os

os.sys.path.insert(0, "/home/schirrmr/code/utils/")
os.sys.path.insert(0, "/home/schirrmr/code/lossy/")
os.sys.path.insert(0, "/home/schirrmr/code/nfnets/")
#os.sys.path.insert(0, "/home/schirrmr/code/invertible-neurips/")
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
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/unfolded-grad/stripes/", #before rebuttal without "icml-"
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
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
    #         'saved_model_folder': '/home/schirrmr/data/exps/lossy/cifar10-wide-nfnets/114/'
    #     }
    # ]


    data_params = dictlistprod({
        'dataset': ['stripes' ],#, 'mnist', 'fashionmnist', 'svhn'],
        'mimic_cxr_target': [None],
        #'dataset': ['cifar10'],#, 'mnist'],
        'saved_model_folder': [None],
    })

    train_params = dictlistprod(
        {
            #"n_epochs": [20],
            "n_epochs": [100],
            "batch_size": [32],
            "train_orig": [False],
            "train_simclr_orig": [False],
            "train_ssl_orig_simple": [False],
            "ssl_loss_factor": [None],
            "loss_name": ['grad_act_match'],
            "grad_from_orig": [True],
        }
    )

    noise_params = [{
            "noise_augment_level": 0,
            'trivial_augment': False,
            'extra_augs': False,
        },
    ]

    quantize_params = [
        {
            "noise_after_simplifier": True,
            "noise_before_generator": False,
            "np_th_seed": 0,
            'quantize_after_simplifier': True,
        }
    ]


    random_params = dictlistprod(
        {
            #"np_th_seed": [0],
        }
    )

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
            'save_models': [True],
            'activation': ["shifted_softplus_1"],
        }
    )
    optim_params = dictlistprod(
        {
            "resample_augmentation": [False],# default this was True
            "resample_augmentation_for_clf": [False], # default this was False
            "std_aug_magnitude": [None],#0.25
            "weight_decay": [1e-5],
            "lr_clf": [5e-4],#5e-4,
            "lr_preproc": [5e-4],
            "threshold": [
                0.1,
            ],
            "optim_type": [
                "adamw",
            ],
            "bpd_weight": [0,1,2,4,],#[0., 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0],
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            train_params,
            random_params,
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
