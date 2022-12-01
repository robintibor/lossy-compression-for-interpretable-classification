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


    train_params = dictlistprod(
        {
            "separate_orig_clf": [False],
            "dist_name": ["normed_sse",],
            "train_clf_on_dist_loss": [False],
            "train_clf_on_orig_simultaneously": [False],
            "orig_loss_weight": [0],
            "stop_clf_grad_through_simple": [False],
            "simple_clf_loss_weight": [0,],
            "frozen_clf": [True],
            "n_epochs": [3],
            "simple_orig_pred_loss_weight": [0],
            "dist_threshold": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],  # ],#0.05,
            "noise_augment_level": [0],
        })

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
                # "n_pretrain_preproc_epochs": [0,1],
            }
        )
        + dictlistprod(
            {
                "lr_preproc": [1e-4,],
                "preproc_name": ["glow_with_pure_resnet"],#res_unet
                "cat_clf_chans_for_preproc": [False, ],#, True],
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


    data_params = dictlistprod(
        {
            # "dataset": ['cifar10'],
            # 610 better..
            # "saved_model_folder": [
            #    '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/537/',
            #    '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/610/'
            # ],
        }) + dictlistprod(
        {
            "dataset": ['svhn'],
            # 610 better..
            "saved_model_folder": [
                '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/383/',
            #    '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/610/'
            ],
        })
    

    model_params = dictlistprod({
        #"/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
        "glow_model_path_32x32": [
            "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
        ],
    })

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            train_params,
            preproc_params,
            data_params,
            model_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params
