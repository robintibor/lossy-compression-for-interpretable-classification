import json
import logging
import os.path
import sys

import kornia
import numpy as np
import pandas as pd
import torch
import torch as th
from torchvision import transforms

from lossy.modules import Expression
from lossy.util import set_random_seeds
from tensorboardX.writer import SummaryWriter
from torch import nn

from lossy import data_locations
from lossy.datasets import get_dataset
from lossy.image_convert import img_0_1_to_glow_img
from lossy.util import np_to_th, th_to_np

from kornia.filters import GaussianBlur2d
from lossy.simclr import compute_nt_xent_loss, modified_simclr_pipeline_transform
from lossy.invglow.invertible.actnorm import ActNorm
from lossy.invglow.invertible.affine import AffineCoefs, AffineModifier, AdditiveCoefs
from lossy.invglow.invertible.branching import ChunkChans, ChunkByIndices
from lossy.invglow.invertible.coupling import CouplingLayer
from lossy.invglow.invertible.distribution import Unlabeled, NClassIndependentDist
from lossy.invglow.invertible.graph import CatChansNode
from lossy.invglow.invertible.graph import Node, SelectNode, CatAsListNode
from lossy.invglow.invertible.graph import get_nodes_by_names
from lossy.invglow.invertible.identity import Identity
from lossy.invglow.invertible.inv_permute import InvPermute, Shuffle
from lossy.invglow.invertible.sequential import InvertibleSequential
from lossy.invglow.invertible.split_merge import ChunkChansIn2, EverySecondChan
from lossy.invglow.invertible.splitter import SubsampleSplitter
from lossy.invglow.invertible.view_as import Flatten2d, ViewAs
from lossy.invglow.models.glow import flow_block
from lossy.datasets import ImageNet
from lossy.image_convert import img_0_1_to_glow_img
from lossy.modules import Expression
from invglow.invertible.init import init_all_modules
from lossy.optim import grads_all_finite

log = logging.getLogger(__name__)

def create_glow_model(
        hidden_channels,
        K,
        L,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        n_chans,
        block_type='conv',
        use_act_norm=True,
        image_size=32,
):
    image_shape = (image_size, image_size, n_chans)

    H, W, C = image_shape
    flows_per_scale = []
    act_norms_per_scale = []
    dists_per_scale = []
    for i in range(L):
        C, H, W = C * 4, H // 2, W // 2

        splitter = SubsampleSplitter(
            2, via_reshape=True, chunk_chans_first=True, checkerboard=False,
            cat_at_end=True)

        if block_type == 'dense':
            pre_flow_layers = [Flatten2d()]
            in_channels = C * H * W
        else:
            assert block_type == 'conv'
            pre_flow_layers = []
            in_channels = C

        flow_layers = [flow_block(in_channels=in_channels,
                                  hidden_channels=hidden_channels,
                                  flow_permutation=flow_permutation,
                                  flow_coupling=flow_coupling,
                                  LU_decomposed=LU_decomposed,
                                  cond_channels=0,
                                  cond_merger=None,
                                  block_type=block_type,
                                  use_act_norm=use_act_norm) for _ in range(K)]

        if block_type == 'dense':
            post_flow_layers = [ViewAs((-1, C * H * W), (-1, C, H, W))]
        else:
            assert block_type == 'conv'
            post_flow_layers = []
        flow_layers = pre_flow_layers + flow_layers + post_flow_layers
        flow_this_scale = InvertibleSequential(splitter, *flow_layers)
        flows_per_scale.append(flow_this_scale)

        if i < L - 1:
            # there will be a chunking here
            C = C // 2
        # act norms for distribution (mean/std as actnorm isntead of integrated
        # into dist)
        act_norms_per_scale.append(InvertibleSequential(Flatten2d(),
                                                        ActNorm((C * H * W),
                                                                scale_fn='exp')))
        dists_per_scale.append(Unlabeled(
            NClassIndependentDist(1, C * H * W, optimize_mean_std=False)))


    dist_nodes = []
    nd_cur = None
    for i in range(L):
        if i > 0:
            nd_cur = SelectNode(nd_cur, 1, name=f'm0-in-flow-{i}')
        nd_cur = Node(nd_cur, flows_per_scale[i], name=f'm0-flow-{i}')
        if i < (L - 1):
            nd_cur = Node(nd_cur, ChunkChans(2), name=f'm0-flow-{i}')
            nd_cur_out = SelectNode(nd_cur, 0)
        else:
            # at last scale, there is no further splitting off of dimensions
            nd_cur_out = nd_cur
        nd_cur_act = Node(nd_cur_out, act_norms_per_scale[i], name=f'm0-act-{i}')
        nd_cur_dist = Node(nd_cur_act, dists_per_scale[i], name=f'm0-dist-{i}')
        dist_nodes.append(nd_cur_dist)

    model = CatAsListNode(dist_nodes, name='m0-full')
    return model


def run_exp(
    output_dir,
    first_n,
    np_th_seed,
    n_epochs,
    lr,
    weight_decay,
    debug,
    hidden_channels,
    L,
    flow_coupling,
):

    writer = SummaryWriter(output_dir)
    hparams = {k: v for k, v in locals().items() if v is not None}
    writer = SummaryWriter(output_dir)
    writer.add_hparams(hparams, metric_dict={}, name=output_dir)
    writer.flush()
    tqdm = lambda x: x
    trange = range

    set_random_seeds(np_th_seed, True)


    log.info("Load data...")

    batch_size = 16
    root = "/data/datasets/ImageNet/imagenet-pytorch"
    # https://github.com/pytorch/examples/blob/fcf8f9498e40863405fe367b9521269e03d7f521/imagenet/main.py#L213-L237
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # load the dataset
    train_dataset = ImageNet(
        root=root,
        split="train",
        transform=train_transform,
        ignore_archive=True,
    )

    if first_n is not None:
        train_dataset = th.utils.data.Subset(train_dataset, np.arange(first_n))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    log.info("Create Glow Model...")

    K = 6
    flow_permutation = 'invconv'
    LU_decomposed = True
    n_chans = 3

    glow = create_glow_model(
        hidden_channels,
        K,
        L,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        n_chans,
        block_type='conv',
        use_act_norm=True,
        image_size=224)

    glow = glow.cuda()
    gen = nn.Sequential(Expression(img_0_1_to_glow_img), glow)
    warmup_steps = 100

    class WrapGen(nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.gen = gen

        def forward(self, x, fixed):
            return self.gen.forward(x + (th.rand_like(x) * 1 / 256.0))

    init_all_modules(WrapGen(gen), train_loader, n_batches=2)

    lr = 5e-4
    weight_decay = 5e-5
    optimizer = th.optim.AdamW(
        [p for p in gen.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    if (warmup_steps is not None) and (warmup_steps > 0):
        lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup_steps)  # noqa
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    i_batch = 0
    for i_epoch in range(n_epochs):
        for X, _ in tqdm(train_loader):
            X = X.cuda()
            z, lp = gen(X)
            nll_loss = -th.mean(lp)

            optimizer.zero_grad(set_to_none=True)
            nll_loss.backward()
            if grads_all_finite(optimizer):
                optimizer.step()
                scheduler.step()
            else:
                print("Grads not finite!!")
            optimizer.zero_grad(set_to_none=True)
            bpd = th_to_np((nll_loss + np.log(256) * np.prod(X.shape)) / (np.prod(X.shape) * np.log(2)))
            writer.add_scalar("bpd", bpd, i_batch)
            results = dict(bpd= bpd)
            writer.flush()
            sys.stdout.flush()
            i_batch += 1
        # later if not debug
        th.save(gen.state_dict(), os.path.join(output_dir, f"gen_state_dict_{i_epoch}.th"))
        th.save(gen, os.path.join(output_dir, f"gen_{i_epoch}.th"))
        th.save(optimizer.state_dict(), os.path.join(output_dir, f"opt_state_dict_{i_epoch}.th"))


    return results
