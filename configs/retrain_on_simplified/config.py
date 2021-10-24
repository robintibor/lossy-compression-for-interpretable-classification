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
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/retrain-simplified/",
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]
    parent_exp_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/one-step-noise-fixed/'
    df = load_data_frame(parent_exp_folder)
    df = df[(df.finished == True) & (df.debug == False)]
    exp_ids = df[df.n_epochs == 100].index

    exp_params = dictlistprod({
        'parent_exp_folder': ['/work/dlclarge2/schirrmr-lossy-compression/exps/one-step-noise-fixed/'],
        'exp_id': exp_ids,
    })

    train_params = dictlistprod(
        {
            "n_epochs": [100],
            "init_pretrained_clf": [False],#[False, True],
        }
    )

    random_params = dictlistprod(
        {
            "np_th_seed": range(1),
        }
    )

    model_params = dictlistprod(
        {
            'save_models': [True],
            "with_batchnorm": [False],
            "noise_on_simplifier": [True],
        }
    )
    optim_params = dictlistprod(
        {
            "weight_decay": [1e-5],
            "lr_clf": [5e-4],#5e-4,
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            exp_params,
            train_params,
            random_params,
            debug_params,
            model_params,
            optim_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    parent_exp_folder,
    exp_id,
    n_epochs,
    init_pretrained_clf,
    lr_clf,
    np_th_seed,
    weight_decay,
    debug,
    save_models,
    with_batchnorm,
    noise_on_simplifier,
):
    if debug:
        n_epochs = 3
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
    from lossy.experiments.retrain_on_simplified.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
