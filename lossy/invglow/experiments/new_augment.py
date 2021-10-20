# PARENTCONFIG: /home/schirrmr/code/invertible-neurips/invglow/experiments/config.py

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
        'save_folder': '/home/schirrmr/data/exps/invertible-neurips/new-augment/',
    },
    ]

    train_params = dictlistprod({
        'warmup_steps': [None],
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,3),
    })

    data_params = dictlistprod({
        'reflect_pad': [True],
        'dataset': ['cifar10', 'cifar100', 'svhn', ], #'#'
        'n_epochs': [250],
    }) + dictlistprod({
        'reflect_pad': [True],
        'dataset': ['tiny', ], #'#'
        'n_epochs': [10],
    })


    model_params = dictlistprod({
        'saved_model_path': [None],
    })

    grid_params = product_of_list_of_lists_of_dicts([
        save_params,
        train_params,
        data_params,
        model_params,
        random_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params

