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
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/post-hoc-class-grad-match-row-images/",
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    data_params = dictlistprod({
        'image_inds':  [[i] for i in [10,289,293,343,367,1075,1307,1346,1356,2383, 2589,2968]],
    })
    #273,1057

    model_params = dictlistprod({
        "model_name": ['resnet50', 'wide_resnet50_2'],
        "softplus_beta": [4],
            #["/work/dlclarge2/schirrmr-lossy-compression/exps/rebuttal/one-step/22/"],
    })

    train_params = dictlistprod({
        'n_epochs': [750],
        "bpd_weight": [10,20,40,80],
        "val_fn_name": ["relued", "relu_match"],
        "ref_from_orig": [True, False],
    })

    random_params = dictlistprod(
        {
            "np_th_seed": range(1),
        }
    )
    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            train_params,
            data_params,
            random_params,
            model_params,
            debug_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(ex, model_name, n_epochs, image_inds, bpd_weight, np_th_seed, val_fn_name, ref_from_orig, debug,
        softplus_beta):
    if debug:
        n_epochs = 3
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


    from lossy.experiments.post_hoc_class_grad_match import  run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
