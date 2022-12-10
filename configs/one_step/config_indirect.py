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
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/",
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]


    train_params = dictlistprod({
            "separate_orig_clf": [True],
            "dist_name": ["normed_sse",],#"cosine_distance"], #""
            "train_clf_on_dist_loss": [False],
            "train_clf_on_orig_simultaneously": [False],
            "orig_loss_weight": [0],
            "stop_clf_grad_through_simple": [False],
            "simple_clf_loss_weight": [0,],
            "lr_clf": [3e-4],
            "n_epochs": [100],
    })
    
    data_model_params = dictlistprod({
        "dataset":  ['cifar10', 'mnist', 'fashionmnist', 'svhn'],
        "saved_model_folder": [None],
    })

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            train_params,
            data_model_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params
