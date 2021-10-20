import os
import os.path
os.environ['pytorch_data'] = '/home/schirrmr/data/pytorch-datasets/'
os.environ['tiny_data'] = '/home/schirrmr/data/tiny-images/'
os.environ['lsun_data'] = '/home/schirrmr/data/lsun/'
os.environ['brats_data'] = '/home/schirrmr/data/brats-2018//'
os.environ['celeba_data'] = '/data/schirrmr/schirrmr/celeba/CELEB_32/'
os.environ['tiny_imagenet_data'] = '/home/schirrmr/data/tiny-imagenet-200/'
os.sys.path.insert(0, '/home/schirrmr/code/invertible-neurips/')

import pandas as pd
import json
from glob import glob

import numpy as np
from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.results import load_data_frame

from invglow.evaluate import compute_auc_for_scores
import logging

log = logging.getLogger(__name__)
log.setLevel('INFO')



def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
        'save_folder': '/home/schirrmr/data/exps/invertible-neurips/simclr/aucs-no-simclr-4/',
    },
    ]


    folder = '/home/schirrmr/data/exps/invertible-neurips/simclr/'
    df = load_data_frame(folder)
    df = df[df.dataset != 'tiny']
    df = df.fillna('-')
    df = df[(df.debug == False) & (df.finished == True)]

    exp_params = dictlistprod({
        'folder': [folder], #'#'
        'exp_id': np.array(df.index),
        'base_model': ['no_simclr']#'simclr_matching',
    })


    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        exp_params
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run(ex, folder, exp_id, base_model):
    exp_folder = os.path.join(folder, str(exp_id))
    config = json.load(open(os.path.join(
        exp_folder, 'config.json'), 'r'))

    dataset = config['dataset']

    assert dataset != 'tiny'

    itd_set = 'test_' + dataset

    result_files = glob(
        os.path.join(exp_folder, '*.all.results.npy'))

    result_file = sorted(
        result_files,
        key=lambda f: int(os.path.split(f)[-1].split('_')[0]))[-1]


    # Get base exp folder
    simclr_mask_type = config['simclr_mask_type']
    simclr_loss = config['simclr_loss']
    if simclr_loss and base_model == 'simclr_matching':
        folder = '/home/schirrmr/data/exps/invertible-neurips/simclr/'
        df = load_data_frame(folder)
        df = df.fillna('-')
        df = df[(df.debug == False)]
        simclr_mask_type_df = simclr_mask_type or '-'

        this_df = df[
            (df.dataset == 'tiny') &
            (df.finished == True) &
            (df.simclr_mask_type == simclr_mask_type_df) &
            (df.simclr_weight == config['simclr_weight'])]

        assert len(this_df) == 1, f"length of this df {len(this_df)}"

        base_exp_folder = os.path.join(folder, str(this_df.index[0]))

        base_model_files = glob(os.path.join(base_exp_folder, '*_model.th'))
        base_model_file = sorted(
            base_model_files,
            key=lambda f: int(os.path.split(f)[-1].split('_')[0]))[-1]
    else:
        assert (not simclr_loss) or base_model == 'no_simclr'
        if config['reflect_pad']:
            base_model_file = '/home/schirrmr/data/exps/invertible-neurips/new-augment/49/10_model.th'
        else:
            base_model_file = '/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th'

    base_results_file = base_model_file + '.all.results.npy'

    fine_loaders_to_results = np.load(result_file, allow_pickle=True).item()
    base_loaders_to_results = np.load(base_results_file, allow_pickle=True).item()

    # each one compute aucs as before
    result_df = pd.DataFrame()
    for ood_set in fine_loaders_to_results.keys():
        if ood_set == itd_set:
            continue

        base_lps_ood = base_loaders_to_results[ood_set]
        fine_lps_ood = fine_loaders_to_results[ood_set]
        base_lps_itd = base_loaders_to_results[itd_set]
        fine_lps_itd = fine_loaders_to_results[itd_set]

        res = dict(ood_set=ood_set)
        metrics = set(fine_lps_ood.keys()) - set(['bpd'])
        for metric in metrics:
            auc = compute_auc_for_scores(
                fine_lps_ood[metric], fine_lps_itd[metric])
            res[f'auc_{metric}'] = auc
            if not ((base_model == 'no_simclr') and
                    (metric == 'lp_masked_z2')):
                auc_diff = compute_auc_for_scores(
                    fine_lps_ood[metric] - base_lps_ood[metric],
                    fine_lps_itd[metric] - base_lps_itd[metric])
                res[f'auc_{metric}_diff'] = auc_diff
        result_df = result_df.append(res, ignore_index=True, sort=False)
        result_df = result_df[list(res.keys())]
    result_df

    results = dict(config).copy()
    results.pop('seed')
    results['base_results_file'] = base_results_file
    for index, row in result_df.iterrows():
        row = dict(row).copy()
        ood_set = row.pop('ood_set')
        for metric in row:
            results[f'{ood_set}_{metric}'] = row[metric]

    for key, val in results.items():
        ex.info[key] = val