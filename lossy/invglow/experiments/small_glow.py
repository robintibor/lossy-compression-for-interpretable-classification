# PARENTCONFIG: /home/schirrmr/code/invertible-neurips/invglow/experiments/config.py

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
        'save_folder': '/home/schirrmr/data/exps/invertible-neurips/smaller-glow/',
    },
    ]

    train_params = dictlistprod({
        'warmup_steps': [None],
    })

    random_params= dictlistprod({
        'np_th_seed': range(0,1),
    })

    data_params = dictlistprod({
        'shuffle_crop_size': [None],
        'augment': [True,],
    })

    dataset_params = [{
        'dataset': 'cifar100',
        'n_epochs': 250,
        },{
        'dataset': 'tiny',
        'n_epochs': 10,
        },
    ]

    reflect_pad_params = [{
        'reflect_pad': True,
        },
    ]


    model_params = dictlistprod({
        'saved_model_path': [None],
        'saved_optimizer_path': [None],
        'K': [6,],#4,8
        'flow_coupling': ['affine', 'additive'],
        'local_patches': [False],
        'block_type': ['conv'],
        'flow_permutation': ['invconv'],
        'LU_decomposed': [True],
        'joint_first_scales': [False],
    })


    grid_params = product_of_list_of_lists_of_dicts([
        reflect_pad_params,
        save_params,
        train_params,
        data_params,
        random_params,
        model_params,
        dataset_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params

