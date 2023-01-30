import os
os.sys.path.insert(0, '/home/schirrmr/code/utils/')
os.sys.path.insert(0, '/home/schirrmr/code/lossy/')
os.sys.path.insert(0, '/home/schirrmr/code/nfnets/')
os.sys.path.insert(0, '/home/schirrmr/code/cifar10-clf/')
os.sys.path.insert(0, '/home/schirrmr/code/sam-optim/')
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
        'save_folder': '/work/dlclarge2/schirrmr-lossy-compression/exps/tmlr/nf-net-stripes/',
    }]

    debug_params = [{
        'debug': False,
    }]

    train_params = dictlistprod({
        'n_epochs': [100],
        'adaptive_gradient_clipping': [False],
        'optim_type': ['adamw'],
        'lr': [5e-3,],#[1e-1,5e-2,1e-2,5e-3,3e-4],
        'weight_decay': [1e-5],
    })

    data_params = dictlistprod({
        'split_test_off_train': [False],
        'first_n': [None],
        "dataset": ['mimic-cxr'],
        "mimic_cxr_target": ["pleural_effusion"]
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,1),
    })

    model_params = dictlistprod({
        'model_name': ['wide_nf_net', ],#False
        'depth': [16],
        'width': [2],#2
        'dropout': [0.3],
        'save_model': [True],
        'activation': ["shifted_softplus_1", ],
        'pooling': ['avgpooling'],
        'norm': ['none'],
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
        model_name,
        np_th_seed,
        n_epochs,
        adaptive_gradient_clipping,
        optim_type,
        lr,
        weight_decay,
        depth,
        width,
        dropout,
        activation,
        save_model,
        dataset,
        pooling,
        norm,
        debug,
        mimic_cxr_target,):
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

    import os
    os.environ['pytorch_data'] = '/home/schirrmr/data/pytorch-datasets/'
    os.environ['mimic_cxr'] = "/work/dlclarge2/schirrmr-mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
    os.environ['small_glow_path'] = "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th"
    os.environ['normal_glow_path'] = "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
    os.environ['imagenet'] = "/data/datasets/ImageNet/imagenet-pytorch/"



    from lossy.experiments.regular_train.run import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
