import os
os.sys.path.insert(0, '/home/schirrmr/code/utils/')
os.sys.path.insert(0, '/home/schirrmr/code/lossy/')
os.sys.path.insert(0, '/home/schirrmr/code/nfnets/')
os.sys.path.insert(0, '/home/schirrmr/code/cifar10-clf/')
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
        'save_folder': '/home/schirrmr/data/exps/lossy/mnist-wide-nfnets/',
    }]

    debug_params = [{
        'debug': False,
    }]

    train_params = dictlistprod({
        'n_epochs': [200],
        'adaptive_gradient_clipping': [False],
        'optim_type': ['adamw'],
        'lr': [1e-3],
        'weight_decay': [1e-5],
    })

    data_params = dictlistprod({
        'split_test_off_train': [False],
        'first_n': [None],
        "dataset": ['mnist', 'fashionmnist']
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,1),
    })

    model_params = dictlistprod({
        'nf_net': [True, ],#False
        'depth': [16],
        'widen_factor': [2],
        'dropout': [0.3],
        'save_model': [True],
        'activation': ["relu"],

    })

    optim_params = dictlistprod({
    })
    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        data_params,
        train_params,
        random_params,
        debug_params,
        model_params,
        optim_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
        ex,
        first_n,
        split_test_off_train,
        nf_net,
        np_th_seed,
        n_epochs,
        adaptive_gradient_clipping,
        optim_type,
        lr,
        weight_decay,
        depth,
        widen_factor,
        dropout,
        activation,
        save_model,
        dataset,
        debug,):
    if debug:
        n_epochs = 3
        first_n = 1024
    else:
        first_n = None
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
    from lossy.experiments.wide_nfnet.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
