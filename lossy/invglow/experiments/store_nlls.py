import os
import os.path
os.environ['pytorch_data'] = '/home/schirrmr/data/pytorch-datasets/'
os.environ['tiny_data'] = '/home/schirrmr/data/tiny-images/'
os.environ['lsun_data'] = '/home/schirrmr/data/lsun/'
os.environ['brats_data'] = '/home/schirrmr/data/brats-2018//'
os.environ['celeba_data'] = '/data/schirrmr/schirrmr/celeba/CELEB_32/'
os.environ['tiny_imagenet_data'] = '/home/schirrmr/data/tiny-imagenet-200/'
os.sys.path.insert(0, '/home/schirrmr/code/invertible-neurips/')

import json
from functools import partial
from glob import glob

import numpy as np
import torch as th
from torchvision import transforms
from tqdm import tqdm
from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.results import load_data_frame, remove_columns_with_same_value


from invglow.datasets import PreprocessedLoader
from invglow.datasets import load_train_test_with_defaults
from invglow.datasets import to_tiny_image
from invglow.evaluate import get_rgb_loaders
from invglow.invertible.expression import Expression
from invglow.invertible.gaussian import get_gaussian_log_probs
from invglow.invertible.graph import IntermediateResultsNode
from invglow.invertible.graph import get_nodes_by_names
from invglow.invertible.noise import UniNoise
from invglow.invertible.pure_model import NoLogDet
from invglow.simclr import ApplyMultipleTransforms, compute_nt_xent_loss
from invglow.simclr import modified_simclr_pipeline_transform
from invglow.util import set_random_seeds
import logging

log = logging.getLogger(__name__)
log.setLevel('INFO')



def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
        'save_folder': '/home/schirrmr/data/exps/invertible-neurips/simclr/storenll/',
    },
    ]

    debug_params = [{
        'debug': False,
    }]

    folder = '/home/schirrmr/data/exps/invertible-neurips/simclr/'
    df = load_data_frame(folder)
    df = df.fillna('-')
    #df = df[df.dataset == 'tiny']
    df = df[(df.debug == False) & (df.finished == True)]

    exp_params = dictlistprod({
        'folder': [folder], #'#'
        'first_n': [None],
        'exp_id': np.array(df.index),
    })


    grid_params = product_of_list_of_lists_of_dicts([
        debug_params,
        save_params,
        exp_params
    ])

    return grid_params


def sample_config_params(rng, params):
    return params

def get_nlls(loader, wanted_nodes, node_names, mask=None):
    rs = []
    with th.no_grad():
        for x, y in tqdm(loader):
            outs = IntermediateResultsNode(wanted_nodes)(
                x.cuda(),fixed=dict(y=None))
            lps = outs[1]
            fixed_lps = []
            for lp in lps:
                if len(lp.shape) > 1:
                    assert len(lp.shape) == 2
                    n_components = lp.shape[1]
                    lp = th.logsumexp(lp, dim=1) - np.log(n_components)
                fixed_lps.append(lp)
            node_to_lps = dict(zip(node_names, fixed_lps))
            lp0 = node_to_lps['m0-flow-0'] / 2 + node_to_lps['m0-dist-0']
            lp1 = node_to_lps['m0-flow-1'] / 2 + node_to_lps['m0-dist-1'] - \
                  node_to_lps['m0-flow-0'] / 2
            lp2 = node_to_lps['m0-dist-2'] - node_to_lps['m0-flow-1'] / 2
            lpz0 = node_to_lps['m0-dist-0'] - node_to_lps['m0-act-0']
            lpz1 = node_to_lps['m0-dist-1'] - node_to_lps['m0-act-1']
            lpz2 = node_to_lps['m0-dist-2'] - node_to_lps['m0-act-2']
            lp0 = lp0.cpu().numpy()
            lp1 = lp1.cpu().numpy()
            lp2 = lp2.cpu().numpy()
            lpz0 = lpz0.cpu().numpy()
            lpz1 = lpz1.cpu().numpy()
            lpz2 = lpz2.cpu().numpy()
            lprob = lp0 + lp1 + lp2
            lprobz = lpz0+lpz1+lpz2
            bpd = np.log2(256) - ((lprob / np.log(2)) / np.prod(x.shape[1:]))
            res = dict(lp0=lp0, lp1=lp1, lp2=lp2, lprob=lprob, bpd=bpd,
                          lpz0=lpz0, lpz1=lpz1, lpz2=lpz2, lprobz=lprobz,)
            if mask is not None:
                z2 = outs[0][-1]
                masked_z2 = z2 * mask.unsqueeze(0)
                lp_masked_z2 = get_gaussian_log_probs(th.zeros_like(z2[0]), th.zeros_like(z2[0]),
                               masked_z2).cpu().numpy()
                res['lp_masked_z2'] = lp_masked_z2
            rs.append(res)
    full_r = {}
    for key in rs[0].keys():
        full_r[key] = np.concatenate([r[key] for r in rs])
    return full_r


def run(ex, folder, exp_id, first_n, debug):
    if debug:
        first_n = 512
    th.backends.cudnn.benchmark = True
    exp_folder = os.path.join(folder, str(exp_id))
    config = json.load(open(os.path.join(exp_folder, 'config.json'), 'r'))
    dataset = config['dataset']

    model_files = glob(os.path.join(exp_folder, '*_model.th'))

    model_file = sorted(
        model_files,
        key=lambda f: int(os.path.split(f)[-1].split('_')[0]))[-1]

    #model_file = '/home/schirrmr/data/exps/invertible-neurips/new-augment/49/10_model.th'
    #'/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th'
    model = th.load(model_file)

    simclr_mask_type = config.get('simclr_mask_type', None)
    if simclr_mask_type == 'trained':
        n_epochs = 3 if dataset != 'tiny' else 1
        mask_file_name = os.path.join(exp_folder, f'reconstructed_mask_{n_epochs:d}.th')
        if os.path.exists(mask_file_name):
            mask = th.load(mask_file_name)
        else:
            train_loader, _ = load_train_test_with_defaults(
                dataset, first_n=first_n, reflect_pad=config.get('reflect_pad', False),
                augment=False)
            noise_module = NoLogDet(UniNoise(1 / 256.0, center=False))

            simclr_transform = modified_simclr_pipeline_transform(
                hflip=dataset != 'svhn', reflect_pad=config.get('reflect_pad', False))

            if dataset == 'tiny':
                simclr_transform.transform.transforms.insert(0, partial(to_tiny_image, grey=config['grey']))
            # add noise
            simclr_transform.transform.transforms.append(noise_module)
            train_loader.dataset.transform = simclr_transform

            simclr_alphas = th.zeros(768, requires_grad=True, device='cuda')
            optim = th.optim.Adam([dict(
                params=[simclr_alphas],
                lr=5e-3,
                weight_decay=5e-5)])
            simclr_weight = config['simclr_weight']
            model.remove_cur_in_out()
            for i_epoch in range(n_epochs):
                for i_batch, ((xa, xb), y) in enumerate(tqdm(train_loader)):
                    with th.no_grad():
                        za, _ = model(xa[:len(xa) // 2].cuda())
                        zb, _ = model(xb[:len(xb) // 2].cuda())
                        za = za[-1]
                        zb = zb[-1]
                    mask = th.sigmoid(simclr_alphas)
                    za = za * mask.unsqueeze(0)
                    zb = zb * mask.unsqueeze(0)
                    loss = compute_nt_xent_loss(za, zb, temperature=0.5)
                    loss = simclr_weight * loss
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
            mask = th.sigmoid(simclr_alphas).detach()
            th.save(mask, open(mask_file_name, 'wb'))
    elif simclr_mask_type == 'fixed64':
        fixed_simclr_mask = th.zeros(768, device='cuda', requires_grad=False)
        fixed_simclr_mask.data[6::12] = 1
        mask = fixed_simclr_mask
    elif simclr_mask_type == 'fixed128':
        fixed_simclr_mask = th.zeros(768, device='cuda', requires_grad=False)
        fixed_simclr_mask.data[3::6] = 1
        mask = fixed_simclr_mask
    else:
        assert simclr_mask_type is None
        mask = None

    loaders = get_rgb_loaders(first_n=first_n, )

    node_names = ('m0-flow-0', 'm0-act-0', 'm0-dist-0',
                  'm0-flow-1', 'm0-act-1', 'm0-dist-1',
                  'm0-flow-2', 'm0-act-2', 'm0-dist-2')
    try:
        wanted_nodes = get_nodes_by_names(model, *node_names)
    except:
        wanted_nodes = get_nodes_by_names(model.module, *node_names)

    first_n_str = str(first_n) if first_n is not None else "all"
    save_results_file = model_file + f'.{first_n_str}.results.npy'

    if os.path.exists(save_results_file):
        loaders_to_results = np.load(
            save_results_file, allow_pickle=True).item()
    else:
        loaders_to_results = {}
    noise_factor = 1 / 256.0
    for set_name, loader in loaders.items():
        if set_name.startswith('test') and (not
                set_name in loaders_to_results):
            log.info(f"Compute for {set_name} ...")
            set_random_seeds(20191120, True)
            # add half of noise interval
            loader = PreprocessedLoader(
                loader,
                Expression(lambda x: x + noise_factor / 2),
                to_cuda=True)
            result = get_nlls(
                loader, wanted_nodes, node_names, mask=mask)
            loaders_to_results[set_name] = result
        elif set_name in loaders_to_results:
            log.info(f"Skipping {set_name}, already exists")
    np.save(save_results_file, loaders_to_results)


