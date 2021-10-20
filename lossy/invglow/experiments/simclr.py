# PARENTCONFIG: /home/schirrmr/code/invertible-neurips/invglow/experiments/config.py

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
        'save_folder': '/home/schirrmr/data/exps/invertible-neurips/simclr/',
    },
    ]

    train_params = dictlistprod({
        'warmup_steps': [None],
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,1),
    })

    data_params = dictlistprod({
        'dataset': ['cifar10',], #'#'# 'svhn',
        'n_epochs': [250],
    })

    reflect_pad_params = [{
        'reflect_pad': False,
        'saved_base_model_path': '/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th',
        'saved_model_path': '/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th',
        },
        # {
        # 'reflect_pad': True,
        # 'saved_base_model_path': '/home/schirrmr/data/exps/invertible-neurips/new-augment/49/10_model.th',
        # 'saved_model_path': '/home/schirrmr/data/exps/invertible-neurips/new-augment/49/10_model.th',
        # },
    ]

    simclr_params = dictlistprod({
        'simclr_weight': [100,1000,],
        'simclr_loss': [True],
        'simclr_mask_type': ['fixed64', 'fixed128', 'trained', None],
    })



    grid_params = product_of_list_of_lists_of_dicts([
        reflect_pad_params,
        save_params,
        train_params,
        data_params,
        random_params,
        simclr_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params

