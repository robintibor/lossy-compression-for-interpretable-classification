import os

os.sys.path.insert(0, "/home/schirrmr/code/utils/")
os.sys.path.insert(0, "/home/schirrmr/code/lossy/")
os.sys.path.insert(0, "/home/schirrmr/code/nfnets/")
os.sys.path.insert(0, "/home/schirrmr/code/invertible-neurips/")
os.sys.path.insert(0, "/home/schirrmr/code/dataset-condensation//")
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
            "save_folder": "/home/schirrmr/data/exps/lossy/continual/train-svhn/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    train_params = dictlistprod(
        {
            "n_epochs_per_stage": [
                50, #100
            ],
            "weight_decay": [1e-4,],  # 5e-4,
            "assumed_std": [1],#[0.5,0.3,0.2],
            "crop_pad": [0,1,2,3,4],
        }
    )

    random_params = dictlistprod(
        {
            "np_th_seed": range(0, 3),
        }
    )

    optim_params = dictlistprod(
        {
            "lrs": [
              #  [0.1, 0.1],
              #  [0.1, 0.01],
              #  [0.01, 0.01],
              #  [0.1, 0.01, 0.001],
              #  [0.1, 0.1, 0.1],
              #  [0.01, 0.01, 0.01],
                [0.01,0.01],
                [0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01,0.01,0.01],
                ],
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            train_params,
            random_params,
            debug_params,
            optim_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    n_epochs_per_stage,
    lrs,
    np_th_seed,
    weight_decay,
    assumed_std,
    crop_pad,
    debug,
):
    if debug:
        n_epochs_per_stage = 3
        first_n = 1024
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
    from lossy.experiments.svhn.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
