import os

os.sys.path.insert(0, "/home/schirrmr/code/utils/")
os.sys.path.insert(0, "/home/schirrmr/code/lossy/")
os.sys.path.insert(0, "/home/schirrmr/code/nfnets/")
os.sys.path.insert(0, "/home/schirrmr/code/invertible-neurips/")
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
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/rebuttal/l2-simplifier/",
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]


    data_params = dictlistprod({#'cifar10',
        'dataset': ['mnist', 'fashionmnist', 'svhn'],#'mnist', 'fashionmnist', 'svhn'
    })

    train_params = dictlistprod(
        {
            #"n_epochs": [20],
            "n_epochs": [50],
            "batch_size": [32],
        }
    )


    quantize_params = [{
        "noise_after_simplifier": True,
        "noise_before_generator": False,
        'quantize_after_simplifier': False,
    }]

    random_params = dictlistprod(
        {
            "np_th_seed": [0],
        }
    )

    model_params = dictlistprod(
        {
            'save_models': [True],
        }
    )
    optim_params = dictlistprod(
        {
            "lr_preproc": [5e-4],
            "bpd_weight": [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0],#[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "mse_weight": [10],
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
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    n_epochs,
    lr_preproc,
    bpd_weight,
    np_th_seed,
    batch_size,
    dataset,
    debug,
    save_models,
    noise_before_generator,
    noise_after_simplifier,
    quantize_after_simplifier,
    mse_weight,
):
    if debug:
        n_epochs = 3
        first_n = 1024
        save_models = False
    else:
        first_n = None
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
    os.environ['pytorch_data'] = '/home/schirrmr/data/pytorch-datasets/'
    os.environ['mimic_cxr'] = "/work/dlclarge2/schirrmr-mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    os.environ['small_glow_path'] = "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
    os.environ['normal_glow_path'] = "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
    os.environ['imagenet'] = "/data/datasets/ImageNet/imagenet-pytorch/"

    from lossy.experiments.l2_simplifier.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
