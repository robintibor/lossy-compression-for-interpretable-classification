import logging
import os.path

import numpy as np
import torch as th
from braindecode.models.modules import Expression
from torch import nn
from tqdm import tqdm

from lossy.affine import AffineOnChans
from lossy.image2image import WrapResidualIdentityUnet, UnetGenerator
from lossy.image_convert import add_glow_noise_to_0_1
from lossy.image_convert import to_plus_minus_one
from rtsutils.util import th_to_np

log = logging.getLogger(__name__)


def create_simplified_data(loader, preproc, ):
    all_X = []
    all_y = []
    for X, y in tqdm(loader):
        X = X.cuda()
        with th.no_grad():
            simple_X = preproc(X)
        all_X.append(th_to_np(simple_X.cpu()))
        all_y.append(th_to_np(y.cpu()))
    return np.concatenate(all_X), np.concatenate(all_y)


def run_exp(
    parent_exp_folder,
    exp_id,
):
    first_n = None
    exp_folder = os.path.join(parent_exp_folder, str(exp_id))
    import json
    from lossy.datasets import get_dataset
    exp_folder = os.path.join(parent_exp_folder, str(exp_id))

    config = json.load(open(os.path.join(exp_folder, 'config.json'), 'r'))
    noise_augment_level = config['noise_augment_level']
    saved_model_folder = config['saved_model_folder']
    dataset = config['dataset']

    assert config['noise_after_simplifier']
    batch_size = config['batch_size']
    split_test_off_train = False

    log.info("Load data...")
    data_path = "/home/schirrmr/data/pytorch-datasets/"
    (
        channel,
        im_size,
        num_classes,
        class_names,
        trainloader,
        train_det_loader,
        testloader,
    ) = get_dataset(
        dataset.upper(),
        data_path,
        batch_size=batch_size,
        standardize=False,
        split_test_off_train=split_test_off_train,
        first_n=first_n,
    )
    assert config['residual_preproc']
    preproc = WrapResidualIdentityUnet(
        nn.Sequential(
            Expression(to_plus_minus_one),
            UnetGenerator(
                3,
                3,
                num_downs=5,
                final_nonlin=nn.Identity,
                norm_layer=AffineOnChans,
                nonlin_down=nn.ELU,
                nonlin_up=nn.ELU,
            ),
        ),
        final_nonlin=nn.Sigmoid(),
    ).cuda()
    preproc = nn.Sequential(preproc, Expression(add_glow_noise_to_0_1))
    preproc.load_state_dict(th.load(os.path.join(exp_folder, 'preproc_state_dict.th')))

    # without noise
    preproc = preproc[0]
    preproc.eval();

    train_X, train_y = create_simplified_data(train_det_loader, preproc)
    test_X, test_y = create_simplified_data(testloader, preproc)
    log.info("Saving train...")
    np.save(os.path.join(exp_folder, 'simple_train_X.npy'), train_X)
    log.info("Saving test...")
    np.save(os.path.join(exp_folder, 'simple_test_X.npy'), test_X)
    log.info("Saved both.")

