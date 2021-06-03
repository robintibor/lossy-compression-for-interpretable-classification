import os
os.sys.path.insert(0, '/home/schirrmr/code/utils/')
os.sys.path.insert(0, '/home/schirrmr/code/lossy/')
os.sys.path.insert(0, '/home/schirrmr/code/nfnets/')
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
        'save_folder': '/home/schirrmr/data/exps/lossy/cifar10-nfnets/',
    }]

    debug_params = [{
        'debug': False,
    }]

    train_params = dictlistprod({
        'n_epochs': [100],
        'batch_size': [64],
    })

    data_params = dictlistprod({
        'split_test_off_train': [False]
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,3),
    })

    model_params = dictlistprod({
        'n_start_filters': [16,32,64],
        'bias_for_conv': [True],
    })

    optim_params = dictlistprod({
        'lr': [5e-3,1e-2],#'lr': [1e-3, 5e-3,],
        'initialization': ['xavier_normal',],
        'restart_epochs': [100],#[None,100,200],
        'adjust_betas': [True,],
        'zero_init_residual': [False],#Only False?
        'weight_decay': [5e-5],
        'optim_type': ['adam',],
        'n_warmup_epochs': [5],
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
        zero_init_residual,
        initialization,
        restart_epochs,
        adjust_betas,
        lr,
        weight_decay,
        n_epochs,
        batch_size,
        split_test_off_train,
        np_th_seed,
        optim_type,
        n_start_filters,
        bias_for_conv,
        n_warmup_epochs,
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
    from lossy.experiments.nfnet.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
