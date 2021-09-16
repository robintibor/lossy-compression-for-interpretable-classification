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
from hyperoptim.results import load_data_frame
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
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/store-simplified/",
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]
    parent_exp_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/one-step-noise-fixed/'
    df = load_data_frame(parent_exp_folder)
    df = df[(df.finished == True) & (df.debug == False) &
            (df.np_th_seed == 0) & (df.n_epochs == 100) & (df.bpd_weight.isin([0.4, 1.0, 1.6]))]
    exp_ids = df.index

    exp_params = dictlistprod({
        'parent_exp_folder': ['/work/dlclarge2/schirrmr-lossy-compression/exps/one-step-noise-fixed/'],
        'exp_id': exp_ids,
    })


    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            exp_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    parent_exp_folder,
    exp_id,
):
    kwargs = locals()
    kwargs.pop("ex")
    log.setLevel("INFO")
    file_obs = ex.observers[0]
    th.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    start_time = time.time()
    ex.info["finished"] = False
    from lossy.experiments.store_simplified.run import run_exp

    run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True
    ex.info["runtime"] = run_time
