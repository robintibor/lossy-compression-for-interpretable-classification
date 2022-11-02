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
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/icml-rebuttal/large-res-glow/",
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    random_params = dictlistprod(
        {
            "np_th_seed": range(1),
        }
    )

    model_params = dictlistprod(
        {
            "hidden_channels": [128,256],
            "L": [3,4,5]
        }
    )
    optim_params = dictlistprod(
        {
            "weight_decay": [5e-5],
            "lr": [5e-4],#5e-4,
        }
    )

    train_params = dictlistprod(
        {
            "n_epochs": [3],
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            random_params,
            debug_params,
            model_params,
            optim_params,
            train_params
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    np_th_seed,
    n_epochs,
    lr,
    weight_decay,
    debug,
    hidden_channels,
    L,
):
    if debug:
        n_epochs = 3
        first_n = 512
    else:
        first_n = None
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
    os.environ['pytorch_data'] = '/home/schirrmr/data/pytorch-datasets/'
    os.environ['mimic_cxr'] = "/work/dlclarge2/schirrmr-mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    os.environ['small_glow_path'] = "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
    os.environ['normal_glow_path'] = "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
    os.environ['imagenet'] = "/data/datasets/ImageNet/imagenet-pytorch/"

    from lossy.experiments.large_res_glow.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
