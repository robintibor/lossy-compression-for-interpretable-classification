import os
os.sys.path.insert(0, '/home/schirrmr/code/utils/')
os.sys.path.insert(0, '/home/schirrmr/code/lossy/')
os.sys.path.insert(0, '/home/schirrmr/code/nfnets/')
os.sys.path.insert(0, '/home/schirrmr/code/invertible-neurips/')
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
        'save_folder': '/home/schirrmr/data/exps/lossy/cifar10-one-step/',
    }]

    debug_params = [{
        'debug': False,
    }]

    train_params = dictlistprod({
        'n_epochs': [50],
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,1),
    })

    model_params = dictlistprod({
        'n_start_filters': [64],
        'residual_preproc': [True,],
    })
    optim_params = dictlistprod({
        'lr': [5e-4,1e-4],
        'threshold': [0.1,],
        'optim_type': ['adam',],
        'bpd_weight': [0.1, 0.5, 1.0]
    })
    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        train_params,
        random_params,
        debug_params,
        model_params,
        optim_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run(ex, n_epochs, optim_type, n_start_filters, residual_preproc,
            lr, threshold, bpd_weight, np_th_seed, debug):
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
    from lossy.experiments.one_step.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
