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
import pandas as pd
import numpy as np

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s")


log = logging.getLogger(__name__)
log.setLevel("INFO")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
            "save_folder": "/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/cifar10ceval/",
                           #"/home/schirrmr/data/exps/lossy/cifar10-one-step/",
        }
    ]

    # one_step_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/'
    # one_step_normed_sse_df = load_data_frame(one_step_folder)
    # one_step_normed_sse_df = one_step_normed_sse_df[one_step_normed_sse_df.finished == 1]
    # one_step_normed_sse_df = one_step_normed_sse_df.fillna('-')
    # one_step_normed_sse_df = one_step_normed_sse_df[one_step_normed_sse_df.debug == 0]
    # one_step_normed_sse_df = one_step_normed_sse_df.drop('save_folder', axis=1)
    # one_step_normed_sse_df = one_step_normed_sse_df.drop('seed', axis=1)
    # one_step_normed_sse_df.loc[:, 'exp_id'] = one_step_normed_sse_df.index
    #
    # retrain_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/retrain-grad-act-match/'
    # retrain_new_df = load_data_frame(retrain_folder)
    # retrain_new_df = retrain_new_df[retrain_new_df.finished == 1]
    # retrain_new_df = retrain_new_df.fillna('-')
    # retrain_new_df = retrain_new_df[retrain_new_df.debug == 0]
    # retrain_new_df = retrain_new_df.drop('save_folder', axis=1)
    # retrain_new_df = retrain_new_df.drop('seed', axis=1)
    # retrain_new_df.loc[:, 'exp_id'] = [int(os.path.split(f)[-1]) for f in retrain_new_df.saved_exp_folder]
    # retrain_new_df_normed_sse = retrain_new_df[retrain_new_df.saved_exp_folder.str.startswith(
    #     '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/')]
    # retrain_new_df_normed_sse = retrain_new_df_normed_sse.drop('saved_exp_folder', axis=1)
    # merged_df_normed_sse = retrain_new_df_normed_sse[
    #     retrain_new_df_normed_sse.add_original_data == True].join(
    #     one_step_normed_sse_df, on='exp_id', lsuffix='_after', rsuffix='_before')
    #
    # merged_df_normed_sse = merged_df_normed_sse
    # print(len(retrain_new_df_normed_sse))
    # retrain_df = merged_df_normed_sse[merged_df_normed_sse.dataset_before == 'cifar10'].sort_values(
    #     by='dist_threshold', ascending=False)
    #exp_ids = retrain_df.index
    #parent_exp_folder = retrain_folder
    # one_step_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/'
    # one_step_normed_sse_df = load_data_frame(one_step_folder)
    # one_step_normed_sse_df = one_step_normed_sse_df[one_step_normed_sse_df.finished == 1]
    # one_step_normed_sse_df = one_step_normed_sse_df.fillna('-')
    # one_step_normed_sse_df = one_step_normed_sse_df[one_step_normed_sse_df.debug == 0]
    # one_step_normed_sse_df = one_step_normed_sse_df.drop('save_folder', axis=1)
    # one_step_normed_sse_df = one_step_normed_sse_df.drop('seed', axis=1)
    # one_step_normed_sse_df.loc[:, 'runtime'] = pd.to_timedelta(np.round(one_step_normed_sse_df.runtime), unit='s')
    # one_step_normed_sse_df.loc[:, 'exp_id'] = one_step_normed_sse_df.index
    #
    # retrain_folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/retrain-grad-act-match/'
    # retrain_new_df = load_data_frame(retrain_folder)
    # retrain_new_df = retrain_new_df[retrain_new_df.finished == 1]
    # retrain_new_df = retrain_new_df.fillna('-')
    # retrain_new_df = retrain_new_df[retrain_new_df.debug == 0]
    # retrain_new_df = retrain_new_df.drop('save_folder', axis=1)
    # retrain_new_df = retrain_new_df.drop('seed', axis=1)
    # retrain_new_df.loc[:, 'runtime'] = pd.to_timedelta(np.round(retrain_new_df.runtime), unit='s')
    # retrain_new_df.loc[:, 'exp_id'] = [int(os.path.split(f)[-1]) for f in retrain_new_df.saved_exp_folder]
    # retrain_new_df.loc[:, 'retrain_exp_id'] = retrain_new_df.index
    # retrain_new_df_normed_sse = retrain_new_df[retrain_new_df.saved_exp_folder.str.startswith(
    #     '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/')]
    # retrain_new_df_normed_sse = retrain_new_df_normed_sse.drop('saved_exp_folder', axis=1)
    #
    # print(len(retrain_new_df_normed_sse))
    #
    # merged_df_normed_sse = retrain_new_df_normed_sse[
    #     retrain_new_df_normed_sse.add_original_data == True].join(
    #     one_step_normed_sse_df, on='exp_id', lsuffix='_after', rsuffix='_before')
    #
    # assert (merged_df_normed_sse.exp_id_after == merged_df_normed_sse.exp_id_before).all()
    #
    # this_df_normed_sse = merged_df_normed_sse[
    #     (merged_df_normed_sse.dataset_before != 'mimic-cxr') &
    #     (merged_df_normed_sse.preproc_name == 'res_glow_with_pure_resnet') &
    #     (merged_df_normed_sse.train_clf_on_orig_simultaneously == False) &
    #     (merged_df_normed_sse.stripes_factor != '-') &
    #     (merged_df_normed_sse.loss_name == 'grad_act_match')]
    # this_df = this_df_normed_sse[
    #     (this_df_normed_sse.dataset_before == 'cifar10') &
    #     (this_df_normed_sse.train_preproc_bpd != '-') &
    #     (this_df_normed_sse.preproc_glow_path == '-')
    #     ]
    #
    # saved_exp_folders = [os.path.join(retrain_folder, str(exp_id))
    #                      for exp_id in this_df.retrain_exp_id]

    folder = '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/gradactmatch/'
    df = load_data_frame(folder)
    df = df.fillna('-')
    df = df[df.debug == 0]
    df = df[df.finished == True]
    df = df.drop('save_folder', axis=1)
    df = df.drop('seed', axis=1)
    df.loc[:, 'runtime'] = pd.to_timedelta(np.round(np.float32(df.runtime)), unit='s')
    # df = df[df.n_epochs == 100]
    print(len(df))
    # remove_columns_with_same_value(df[df.scale_dists_loss_by_n_vals == False]).sort_values(
    #    by=['bpd_weight', 'loss_name'])
    part_df = df[
        (df.finished == True) &
        (df.n_epochs == 100) &
        (df.skip_unneeded_bpd_computations == True) &
        (df.bpd_weight == 4) &
        (df.dataset == 'cifar10') &
        (df.separate_orig_clf == True) &
        (df.loss_name == "grad_act_match") &
        (df.preproc_glow_path == '-')].sort_values(by='dist_threshold')  # temporarily, until results computed
    saved_exp_folders = [os.path.join(folder, str(exp_id)) for exp_id in part_df.index]

    #saved_exp_folders = [os.path.join(parent_exp_folder, str(exp_id))
    #                     for exp_id in exp_ids]


    exp_params = dictlistprod({
        'saved_exp_folder': saved_exp_folders,
    })



    eval_params = dictlistprod(
        {
            'with_preproc': [True, False],
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            exp_params,
            eval_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    saved_exp_folder,
    with_preproc,
):
    kwargs = locals()
    kwargs.pop("ex")
    log.setLevel("INFO")
    # file_obs = ex.observers[0]
    # output_dir = file_obs.dir
    # kwargs["output_dir"] = output_dir
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


    from lossy.experiments.cifar10ceval import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
