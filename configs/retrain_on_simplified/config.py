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
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/retrain-grad-act-match/",
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]
    # parent_exp_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/rebuttal/one-step/'
    # df = load_data_frame(parent_exp_folder)
    # df = df[(df.finished == True) & (df.debug == False) & (df.np_th_seed == 0)]
    # df = df[(df.quantize_after_simplifier == True) &
    #         (df.noise_after_simplifier == True)]
    # exp_ids = df.index
    #
    # saved_exp_folders = [os.path.join(parent_exp_folder, str(exp_id))
    #                      for exp_id in exp_ids]
    # parent_exp_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/rebuttal/l2-simplifier//'
    # df = load_data_frame(parent_exp_folder)
    # df = df[df.finished == 1]
    # df = df.fillna('-')
    # df = df[df.debug == 0]
    # df = df[(df.mse_weight == 10) & (df.noise_after_simplifier == True) &
    #         (df.bpd_weight > 1.0)]
    # exp_ids = df.index
    # saved_exp_folders = [os.path.join(parent_exp_folder, str(exp_id))
    #      _cl                for exp_id in exp_ids]

    parent_exp_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/'#
    #'/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/unfolded-grad/'#'/work/dlclarge2/schirrmr-lossy-compression/exps/rebuttal/one-step-simclr/'
    df = load_data_frame(parent_exp_folder)
    df = df[df.debug == False]
    df = df[df.finished == True]
    df = df.fillna('-')

    df = df[
        (df.n_epochs == 100) &
        (df.skip_unneeded_bpd_computations == True) &
        (df.dataset == 'cifar10') &
        (df.bpd_weight == 4) &
        (df.separate_orig_clf == True) &
        (df.preproc_glow_path == "-")
        ]

    exp_ids = df.index
    #saved_exp_folders = [os.path.join(parent_exp_folder, str(exp_id))
    #                     for exp_id in exp_ids]

    parent_exp_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/'
    nfnet_exp_ids = [1523, 1533, 1535, 1540, 1541, 1543]
    vit_exp_ids = [1637,1638,1639,1640,1641,1642]
    saved_clf_exp_folders = [
        os.path.join(parent_exp_folder, str(nfnet_exp_ids[0])),
        os.path.join(parent_exp_folder, str(vit_exp_ids[0])),
        ]
    saved_preproc_exp_folders = [os.path.join(parent_exp_folder, str(exp_id))
                             for exp_id in nfnet_exp_ids + vit_exp_ids]


    exp_params = dictlistprod({
        'saved_clf_exp_folder': saved_clf_exp_folders,
        'saved_preproc_exp_folder' : saved_preproc_exp_folders,
    })

    train_params = dictlistprod(
        {
            "n_epochs": [100],
            "init_pretrained_clf": [False,],#[False, True],#True
            "restandardize_inputs": [False],
            "contrast_normalize": [False],
            "add_original_data": [False],
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
            "use_saved_clf_model_folder": [False],
        }
    )
    optim_params = dictlistprod(
        {
            "weight_decay": [1e-5],
            "lr_clf": [5e-4],#5e-4,
        }
    )

    blur_and_jpg_params = [{
        'blur_simplifier': False,
        'blur_sigma': None,
        'dataset': None,
        'jpg_quality': None,
        'simclr_loss_factor': None,
    }]

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            exp_params,
            train_params,
            random_params,
            debug_params,
            model_params,
            optim_params,
            blur_and_jpg_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    saved_preproc_exp_folder,
    n_epochs,
    init_pretrained_clf,
    lr_clf,
    np_th_seed,
    weight_decay,
    debug,
    save_models,
    with_batchnorm,
    restandardize_inputs,
    contrast_normalize,
    add_original_data,
    dataset,
    blur_simplifier,
    blur_sigma,
    jpg_quality,
    simclr_loss_factor,
    use_saved_clf_model_folder,
    saved_clf_exp_folder,
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

    import os
    os.environ['pytorch_data'] = '/home/schirrmr/data/pytorch-datasets/'
    os.environ['mimic_cxr'] = "/work/dlclarge2/schirrmr-mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    os.environ['small_glow_path'] = "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
    os.environ['normal_glow_path'] = "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
    os.environ['imagenet'] = "/data/datasets/ImageNet/imagenet-pytorch/"


    from lossy.experiments.retrain_on_simplified.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
