# PARENTCONFIG: /home/schirrmr/code/invertible-neurips/invglow/experiments/config.py

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
        'save_folder': '/home/schirrmr/data/exps/invertible-neurips/shuffle-crops/',
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
        'shuffle_crop_size': [2,4,8,16],
        'augment': [True, False],
    })

    reflect_pad_params = [{
        'reflect_pad': False,
        'saved_base_model_path': '/home/schirrmr/data/exps/invertible/finetune/12/76_model.neurips.th',
        'saved_model_path': '/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th',
        },
    ]


    grid_params = product_of_list_of_lists_of_dicts([
        reflect_pad_params,
        save_params,
        train_params,
        data_params,
        random_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params

