import os
os.sys.path.insert(0, '/home/schirrmr/code/invertible-public/')
os.sys.path.insert(0, '/home/schirrmr/code/invertible-eeg//')
os.sys.path.insert(0, '/home/schirrmr/code/invertible-neurips///')
os.sys.path.insert(0, '/home/schirrmr/code/utils/')
os.sys.path.insert(0, '/home/schirrmr/code/lossy/')
import time
import logging

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts

import torch as th
import torch.backends.cudnn as cudnn
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s')


log = logging.getLogger(__name__)
log.setLevel('INFO')


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [{
        'save_folder': '/home/schirrmr/data/exps/lossy-matched/cifar10-train-clf-together/',
    }]

    debug_params = [{
        'debug': False,
    }]

    train_params = dictlistprod({
        'n_epochs': [50],
        'train_classifier': [False],
        'train_classifier_together': [True],
        'batch_size': [32],
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,1),
    })

    optim_params = dictlistprod({
        'lr_preproc': [5e-4, 1e-4,],
        'class_loss_factor': [0.75],
        'optim_preproc_name': ['adam', 'radam'],
        'cosine_period': [None, 2083],
        'lr_clf': [5e-4],
    })


    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        train_params,
        random_params,
        debug_params,
        optim_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
        ex,
        lr_preproc,
        np_th_seed,
        debug,
        n_epochs,
        optim_preproc_name,
        class_loss_factor,
        cosine_period,
        train_classifier,
        batch_size,
        train_classifier_together,
        lr_clf,):
    kwargs = locals()
    kwargs.pop('ex')
    if not debug:
        log.setLevel('INFO')
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    kwargs['output_dir'] = output_dir
    th.backends.cudnn.benchmark = True
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False
    from lossy.experiments.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
