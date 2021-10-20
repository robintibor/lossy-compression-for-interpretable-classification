import os

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

    save_params = [
        {
        'save_folder': None, # should be set by subconfig
    },
    ]

    debug_params = [{
        'debug': False,
    }]

    data_params = dictlistprod({
        'dataset': ['cifar10', 'cifar100', ], #'#'
        'first_n': [None],
        'exclude_cifar_from_tiny': [False],
        'noise_factor': [1/256.0],
        'batch_size': [64],
        'augment': [True],
        'grey': [False],
        'shuffle_crop_size': [None],
    })

    train_params = dictlistprod({
        'n_epochs': [250],
        'warmup_steps': [None],
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,2),
    })

    optim_params = dictlistprod({
        'lr': [5e-4],
        'weight_decay': [5e-5],
    })

    model_params = dictlistprod({
        'saved_model_path': [None],#'/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th'],#'/home/rsc7rng/exps/pretrain/14/125_model.th'
        'saved_optimizer_path': [None],
        'reinit': [False,],
        'K': [32],
        'flow_coupling': ['affine'],
        'local_patches': [False],
        'block_type': ['conv'],
        'flow_permutation': ['invconv'],
        'LU_decomposed': [True],
        'joint_first_scales': [False],
    })


    ood_params = dictlistprod({
        'base_set_name': ['tiny'],
        'saved_base_model_path': ['/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th'],
        'outlier_loss': [None],
        'outlier_weight': [None],
        'ood_set_name': ['svhn'],
        'outlier_temperature': [None],
    })
    class_params = [{
        'init_class_model': False,
        'on_top_class_model_name': None,
        'add_full_label_loss': False,
    }]

    simclr_params = [{
        'simclr_weight': None,
        'simclr_loss': False,
        'simclr_mask_type': None,
    }]

    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        data_params,
        train_params,
        debug_params,
        random_params,
        optim_params,
        model_params,
        ood_params,
        class_params,
        simclr_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
        ex,
        dataset,
        first_n,
        lr,
        weight_decay,
        np_th_seed,
        debug,
        n_epochs,
        exclude_cifar_from_tiny,
        saved_base_model_path,
        saved_model_path,
        saved_optimizer_path,
        reinit,
        base_set_name,
        outlier_weight,
        outlier_loss,
        ood_set_name,
        outlier_temperature,
        noise_factor,
        K,
        flow_coupling,
        init_class_model,
        on_top_class_model_name,
        batch_size,
        add_full_label_loss,
        augment,
        grey,
        warmup_steps,
        local_patches,
        block_type,
        flow_permutation,
        LU_decomposed,
        joint_first_scales,
        reflect_pad,
        simclr_mask_type,
        simclr_weight,
        simclr_loss,
        shuffle_crop_size,
):
    kwargs = locals()
    kwargs.pop('ex')
    if not debug:
        log.setLevel('INFO')
    os.environ['pytorch_data'] = '/home/schirrmr/data/pytorch-datasets/'
    os.environ['tiny_data'] = '/home/schirrmr/data/tiny-images/'
    os.environ['lsun_data'] = '/home/schirrmr/data/lsun/'
    os.environ['brats_data'] = '/home/schirrmr/data/brats-2018//'
    os.environ['celeba_data'] = '/home/schirrmr/data/'  # wrong who cares
    os.environ['tiny_imagenet_data'] = '/home/schirrmr/data/'  # wrong who cares

    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    kwargs['output_dir'] = output_dir
    th.backends.cudnn.benchmark = True  # args.benchmark
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False
    from invglow.exp import run_exp

    trainer, _ = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['finished'] = True

    for key, val in trainer.state.results.items():
        ex.info[key] = float(val)
    ex.info['runtime'] = run_time
