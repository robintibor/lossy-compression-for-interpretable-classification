import logging
import os.path
from copy import deepcopy
from functools import partial

import numpy as np
import torch as th
from torch import nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter
from torchvision import transforms

from invglow.load_data import load_data
from invglow.evaluate import compute_bpds, compute_auc_for_scores
from invglow.invertible.categorical_mixture import InvertibleClassConditional
from invglow.invertible.branching import CatChans
from invglow.invertible.distribution import MergeLogDets
from invglow.invertible.graph import CatChansNode
from invglow.invertible.graph import Node
from invglow.invertible.graph import get_nodes_by_names
from invglow.invertible.init import init_all_modules
from invglow.invertible.sequential import InvertibleSequential
from invglow.losses import nll_class_loss
from invglow.models.class_conditional import latent_model
from invglow.models.glow import create_glow_model
from invglow.models.patch_glow import create_patch_glow_model
from invglow.scheduler import ScheduledOptimizer
from invglow.util import check_gradients_clear, step_and_clear_gradients
from invglow.util import grads_all_finite
from invglow.util import np_to_var
from invglow.util import set_random_seeds, var_to_np
from invglow.simclr import modified_simclr_pipeline_transform
from invglow.simclr import compute_nt_xent_loss
from invglow.simclr import ApplyMultipleTransforms
from invglow.datasets import to_tiny_image

import pandas as pd
from invglow.evaluate import compute_bpds, compute_auc_for_scores
from invglow.invertible.gaussian import get_gaussian_log_probs


log = logging.getLogger(__name__)


class ShuffleCrops(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x):
        unfolded = th.nn.functional.unfold(
            x.unsqueeze(0),
            (self.crop_size, self.crop_size),
            stride=(self.crop_size, self.crop_size))
        # Ensure at more than half of positions
        # have been moved
        n_same = unfolded.shape[-1]
        while n_same > (unfolded.shape[-1] // 2):
            permutation = th.randperm(unfolded.shape[-1])
            n_same = th.sum(permutation == th.arange(len(permutation))).item()

        permuted = unfolded[:, :, permutation]
        refolded = th.nn.functional.fold(
            permuted,
            (x.shape[-2], x.shape[-1]),
            (self.crop_size, self.crop_size), stride=(self.crop_size, self.crop_size))
        return refolded[0]

class ParameterHolder(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param


def compute_masked_lpz(model, mask, loader):
    with th.no_grad():
        df = pd.DataFrame()
        for x, _ in loader:
            z, lp = model(x)
            zeros = th.zeros_like(z[2][0])
            log_probs_per_dim = get_gaussian_log_probs(
                zeros, zeros, z[2], sum_dims=False)
            log_probs = th.sum(log_probs_per_dim, dim=1).cpu().numpy()
            masked_lpz = th.sum(mask.unsqueeze(0) * log_probs_per_dim, dim=1).cpu().numpy()
            masked_z = mask.unsqueeze(0) * z[2]
            masked_z_lp = get_gaussian_log_probs(
                zeros, zeros, masked_z, sum_dims=True).detach().cpu().numpy()
            n_dims = np.prod(x.shape[1:])
            bpds = -(lp - np.log(256) * n_dims) / (n_dims * np.log(2))
            batch_df = pd.DataFrame(dict(
                lp=lp.cpu().numpy(),
                lpz2=log_probs, lpz2masked=masked_lpz,
                lpmaskedz2=masked_z_lp,
                bpds=bpds.cpu().numpy()))
            df = pd.concat((df, batch_df))
    return df


class BaseFineIndependent(object):
    """Compute NLLs from base/general model and fine/specific model,
    recompute in case some NLLs are not finite due to numerical instability."""
    def get_nlls(self, model, base_model, x, y, prev_valid_mask=None):
        # remember which nlls were finite (not inf or nan)
        # then recompute forward for only those examples
        valid_mask = np_to_var([True] * len(x), device=x.device)
        n_dims = np.prod(x.shape[1:])
        if base_model is not None:
            with th.no_grad():
                base_nll = -(base_model(
                    x, fixed=dict(y=y))[1] - np.log(256) * n_dims)
                mask = th.isfinite(base_nll.detach())
                valid_mask = mask & valid_mask
        else:
            base_nll = None
        nll = -(model(x, fixed=dict(y=y))[1] - np.log(256) * n_dims)
        # deal with full label multi class pred case
        if len(nll.shape) == 2:
            nll_for_mask = th.sum(nll, dim=1).detach()
        else:
            nll_for_mask = nll.detach()
        mask = th.isfinite(nll_for_mask.detach())
        valid_mask = mask & valid_mask
        n_valid = th.sum(valid_mask)
        if (n_valid < len(x)) and (n_valid > (len(x) // 4)):
            # n valid too small don't handle to prevent too small batch,
            # catch later nans on backward
            del nll
            del base_nll
            del mask
            x = x[valid_mask]
            if y is not None:
                y = y[valid_mask]
            if prev_valid_mask is None:
                prev_valid_mask = valid_mask
            return self.get_nlls(model, base_model, x, y, prev_valid_mask=prev_valid_mask)
        if prev_valid_mask is None:
            prev_valid_mask = valid_mask
        return dict(base_nll=base_nll, fine_nll=nll, valid_mask=prev_valid_mask)


def apply_inlier_losses(
        model, base_model, x,y,  nll_computer,
        add_full_label_loss,
        temperature,
        weight,
        outlier_batches,
        outlier_loss,):
    # Need to compute outputs for all classes for the full label loss
    y_for_outs = None if add_full_label_loss else y
    model_outs = nll_computer.get_nlls(model, base_model, x, y_for_outs)
    if len(model_outs['fine_nll']) < len(y):
        y = y[model_outs['valid_mask']]
    model_outs['masked_y'] = y
    apply_inlier_loss_from_outs(
        model_outs, y,
        add_full_label_loss,
        temperature=temperature, weight=weight,
        outlier_loss=outlier_loss,)
    return model_outs


def apply_inlier_loss_from_outs(
        model_outs,
        y,
        add_full_label_loss,
        temperature,
        weight,
        outlier_loss):
    nll = model_outs['fine_nll']
    total_loss = 0
    if add_full_label_loss:
        if len(y) == len(model_outs['base_nll']):
            assert len(nll.shape) == 2
            # only correct_labels
            nll_for_loss = th.gather(
                nll, dim=1, index=th.argmax(y, dim=1).unsqueeze(1)).squeeze(1)
            assert outlier_loss == 'class'
            # Now compute wrong class loss, sum over all wrong classes
            wrong_class_loss = nll_class_loss(
                model_outs['base_nll'].unsqueeze(1),
                model_outs['fine_nll'],
                target_val=0, temperature=temperature,
                weight=weight, reduction='none')
            # Mask out correct class
            wrong_class_loss = wrong_class_loss * (1 - y.type_as(wrong_class_loss))
            wrong_class_loss = th.mean(wrong_class_loss)
            total_loss = total_loss + wrong_class_loss
        else:
            log.warning("Did not apply label loss because of nonfinite outputs")
            return
    else:
        nll_for_loss = nll
    assert len(nll_for_loss.shape) == 1
    nll_loss = th.mean(nll_for_loss)
    total_loss = total_loss + nll_loss
    total_loss.backward()


def apply_outlier_losses(
        model, base_model, inlier_results,
        ood_x, ood_y,
        nll_computer, temperature, weight, outlier_loss):
    outlier_results = nll_computer.get_nlls(model, base_model, ood_x, ood_y)
    apply_outlier_losses_from_outs(inlier_results, outlier_results,
                                   outlier_loss, temperature=temperature,
                                   weight=weight,
                                   n_dims=np.prod(ood_x.shape[1:]))
    return outlier_results


def apply_outlier_losses_from_outs(
        inlier_results,
        outlier_results,
        outlier_loss,
        temperature,
        weight,
        n_dims):
    if outlier_loss == 'margin':
        n_min = min(len(inlier_results['fine_nll']), len(outlier_results['fine_nll']))
        diff = th.nn.functional.relu(
            n_dims + inlier_results['fine_nll'][:n_min].detach() -
            outlier_results['fine_nll'][:n_min])
        ood_loss = th.mean(diff)
    elif outlier_loss == 'class':
        assert outlier_loss == 'class'
        ood_loss = nll_class_loss(
            outlier_results['base_nll'], outlier_results['fine_nll'],
            target_val=0, temperature=temperature, weight=weight,
            reduction='mean')
    ood_loss.backward()



def run_exp(dataset,
            first_n,
            lr,
            weight_decay,
            np_th_seed,
            debug,
            output_dir,
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
    hidden_channels=512
    L=3
    if debug:
        first_n = 512
        n_epochs = 3
        if dataset == 'tiny':
            # pretrain a bit longer
            first_n = 5120
            n_epochs = 10

    set_random_seeds(np_th_seed, True)

    log.info("Loading data...")
    loaders, base_train_loader = load_data(
        dataset=dataset,
        first_n=first_n,
        exclude_cifar_from_tiny=exclude_cifar_from_tiny,
        base_set_name=base_set_name,
        ood_set_name=ood_set_name,
        noise_factor=noise_factor,
        batch_size=batch_size,
        augment=augment,
        grey=grey,
        reflect_pad=reflect_pad
    )

    n_chans = next(loaders['test'].__iter__())[0].shape[1]

    if saved_model_path is None:
        log.info("Creating model...")
        if not local_patches:
            model = create_glow_model(
                hidden_channels=hidden_channels,
                K=K,
                L=L,
                flow_permutation=flow_permutation,
                flow_coupling=flow_coupling,
                LU_decomposed=LU_decomposed,
                n_chans=n_chans,
                block_type=block_type,
            )
        else:
            model = create_patch_glow_model(
                hidden_channels=hidden_channels,
                K=K,
                flow_permutation=flow_permutation,
                flow_coupling=flow_coupling,
                LU_decomposed=LU_decomposed,
                n_chans=n_chans,
            )
        model = model.cuda();

    if saved_base_model_path is not None:
        log.info("Loading base model...")
        base_model = th.load(saved_base_model_path)
        init_all_modules(base_model, None)
    else:
        base_model = None

    if saved_model_path is not None:
        log.info("Loading pretrained model...")
        model = th.load(saved_model_path)
        if on_top_class_model_name is not None:
            # Check if we have a on top class model already contained
            if hasattr(model, 'sequential') and len(list(model.sequential.children())) == 2:
                log.info("Extracting on top class model...")
                model_log_det_node = model.sequential[0]
                class_model = model.sequential[1].module
                model = InvertibleSequential(
                    model_log_det_node, MergeLogDets(class_model))
                class_model_loaded = True
            else:
                class_model_loaded = False
    else:
        class_model_loaded = False


    if (on_top_class_model_name is not None) and (not class_model_loaded):
        # remove references to previous dist node
        model_log_act_nodes = get_nodes_by_names(
            model, 'm0-act-0', 'm0-act-1', 'm0-act-2')
        for a in model_log_act_nodes:
            a.next = []
        model_log_det_node = CatChansNode(
            model_log_act_nodes,
            notify_prev_nodes=True)

        if on_top_class_model_name == 'latent':
            top_single_class_model = latent_model(n_chans)
        if dataset in ['cifar10', 'svhn', 'fashion-mnist', 'mnist']:
            n_classes = 10
        else:
            assert dataset == 'cifar100'
            n_classes = 100

        i_classes = list(range(n_classes))
        class_model = InvertibleClassConditional(
            [Node(deepcopy(top_single_class_model), CatChans()) for _ in
             i_classes],
            i_classes)
        del top_single_class_model
        class_model.cuda();

        if init_class_model:
            from itertools import islice
            with th.no_grad():
                init_x_y = [(model_log_det_node(x)[0], y) for x, y in
                            islice(loaders['train'], 10)]
                init_x = th.cat([xy[0] for xy in init_x_y], dim=0)
                init_y = th.cat([xy[1] for xy in init_x_y], dim=0)
                init_all_modules(class_model, (init_x, init_y), use_y=True)
        else:
            init_all_modules(class_model, None)

        model = InvertibleSequential(model_log_det_node, MergeLogDets(class_model))
        model.cuda();

    if add_full_label_loss:
        train_inlier_model = InvertibleSequential(model_log_det_node,
                                                  class_model)
        train_inlier_model.cuda();
    else:
        train_inlier_model = model

    if (saved_model_path is None) or reinit:
        init_all_modules(model, loaders['train'], use_y=False)
    else:
        init_all_modules(model, None)

    optimizer = th.optim.Adamax(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)

    if saved_optimizer_path is not None:
        optimizer.load_state_dict(th.load(saved_optimizer_path))

    if (warmup_steps is not None) and (warmup_steps > 0):
        lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup_steps)  # noqa
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        optimizer = ScheduledOptimizer(scheduler, optimizer)


    if (ood_set_name is not None):
        log.info("Compute BPDs of base model...")
        # Remember bpds of base model
        base_model_bpds = {}
        assert base_model is not None
        for loader_name, loader in loaders.items():
            if (base_model is not None):
                bpds = compute_bpds(loader, base_model, use_y=False,
                                    show_tqdm=False)
            base_model_bpds[loader_name] = var_to_np(bpds)
            mean_bpd = th.mean(bpds).item()
            print(f"base_{loader_name} BPD: {mean_bpd:.2f}")

    if outlier_loss is not None:
        train_base_model = base_model
    else:
        train_base_model = None

    nll_computer = BaseFineIndependent()
    train_loader = deepcopy(loaders['train'])

    def get_outlier_batches():
        outlier_batches = []
        x, _ = next(base_train_loader.__iter__())
        outlier_batches.append((x, None))
        return outlier_batches

    # Only need to compute base nll on inliers if needed
    # (full label loss or
    if add_full_label_loss:
        train_base_model_inlier = train_base_model
    else:
        train_base_model_inlier = None

    if joint_first_scales:
        for (n_fine, _), (n_base, _) in zip(model.named_parameters(), base_model.named_parameters()):
            assert n_fine == n_base
            assert n_fine[:6] in ('prev.0', 'prev.1', 'prev.2')
            if n_fine[:6] in ('prev.0', 'prev.1'):
                splitted = n_fine.split('.')
                m_names = splitted[:-1]
                p_name = splitted[-1]
                m_f = model
                m_b = base_model
                for m_n in m_names:
                    m_f = m_f._modules[m_n]
                    m_b = m_b._modules[m_n]
                m_b._parameters[p_name] = m_f._parameters[p_name]
            else:
                assert n_fine[:6] == 'prev.2'
        whole_set_p = set(list(model.parameters()) + list(base_model.parameters()))
        params_to_optimize = [p for p in whole_set_p if p.requires_grad]

        optimizer = th.optim.Adamax(
            params_to_optimize,
            lr=lr, weight_decay=weight_decay)

    # impadd
    if shuffle_crop_size is not None:
        l = train_loader.dataloader
        shuffler = ShuffleCrops(shuffle_crop_size)
        l.dataset.transform.transforms.append(shuffler)

    if simclr_mask_type == 'trained':
        simclr_alphas = th.zeros(
            768, device='cuda', requires_grad=True)
        optimizer.add_param_group(dict(
            params=[simclr_alphas],
            lr=5e-3,
            weight_decay=5e-5))
    elif simclr_mask_type == 'fixed64':
        fixed_simclr_mask = th.zeros(768, device='cuda', requires_grad=False)
        fixed_simclr_mask.data[6::12] = 1
    elif simclr_mask_type == 'fixed128':
        fixed_simclr_mask = th.zeros(768, device='cuda', requires_grad=False)
        fixed_simclr_mask.data[3::6] = 1
    else:
        assert simclr_mask_type is None

    if simclr_loss:
        simclr_transform = modified_simclr_pipeline_transform(
            hflip=dataset != 'svhn', reflect_pad=reflect_pad)
        if dataset == 'tiny':
            simclr_transform.transform.transforms.insert(0, partial(to_tiny_image, grey=grey))
        # add noise
        simclr_transform.transform.transforms.append(train_loader.module)
        old_transforms = train_loader.dataloader.dataset.transform.transforms
        old_with_noise = transforms.Compose(old_transforms + [train_loader.module])
        train_loader = train_loader.dataloader
        comb_transform = ApplyMultipleTransforms(old_with_noise, simclr_transform)
        train_loader.dataset.transform = comb_transform


    def train(engine, batch):
        x, y = batch
        if simclr_loss:
            (x, (xa, xb)) = x
            x = x.cuda()

        check_gradients_clear(optimizer)
        model.train()

        if (outlier_loss is not None) or joint_first_scales:
            outlier_batches = get_outlier_batches()
        else:
            outlier_batches = None
        inlier_results = apply_inlier_losses(
            train_inlier_model,
            train_base_model_inlier,
            x,
            y,
            nll_computer,
            add_full_label_loss=add_full_label_loss,
            temperature=outlier_temperature,
            weight=outlier_weight,
            outlier_loss=outlier_loss,
            outlier_batches=outlier_batches,)

        if simclr_loss:
            # make similarity
            za, _ = model(xa[:len(xa) // 2].cuda())
            zb, _ = model(xb[:len(xb) // 2].cuda())
            za = za[-1]
            zb = zb[-1]
            if simclr_mask_type is not None:
                if simclr_mask_type == 'trained':
                    mask = th.sigmoid(simclr_alphas)
                else:
                    assert simclr_mask_type.startswith('fixed')
                    mask = fixed_simclr_mask.detach()
                za = za * mask.unsqueeze(0)
                zb = zb * mask.unsqueeze(0)
            loss = compute_nt_xent_loss(za, zb, temperature=0.5)
            loss = simclr_weight * loss
            loss.backward()

        if joint_first_scales:
            for out_x, out_y in outlier_batches:
                _, lp = base_model(out_x)
                nll_loss = -th.mean(lp)
                nll_loss.backward()

        if outlier_loss is not None:
            for out_x, out_y in outlier_batches:
                apply_outlier_losses(model, train_base_model,
                                     inlier_results, out_x, out_y,
                                     nll_computer,
                                     outlier_loss=outlier_loss,
                                     temperature=outlier_temperature,
                                     weight=outlier_weight,)
        if grads_all_finite(optimizer):
            step_and_clear_gradients(optimizer)
        else:
            log.warning("NaNs or Infs in grad! Not all grads finite")
            optimizer.zero_grad()
        n_dims = np.prod(x.shape[1:])
        bpd = th.mean(inlier_results['fine_nll']).item() / (n_dims * np.log(2))
        return dict(bpd=bpd)

    eval_model = model

    def evaluate(engine, ):
        with th.no_grad():
            eval_model.eval()
            results = {}
            print(f"Epoch {engine.state.epoch:d}")

            if joint_first_scales:
                base_model_bpds.clear()
                assert base_model is not None
                for loader_name, loader in loaders.items():
                    if (base_model is not None):
                        bpds = compute_bpds(loader, base_model, use_y=False,
                                            show_tqdm=False)
                    base_model_bpds[loader_name] = var_to_np(bpds)
                    mean_bpd = th.mean(bpds).item()
                    print(f"base_{loader_name} BPD: {mean_bpd:.2f}")
                    results[f"base_{loader_name}_bpd"] = mean_bpd
                    writer.add_scalar(f"base_{loader_name}_bpd", mean_bpd,
                                      engine.state.epoch)

            if simclr_mask_type is not None:
                if simclr_mask_type == 'trained':
                    mask = th.sigmoid(simclr_alphas)
                else:
                    assert simclr_mask_type.startswith('fixed')
                    mask = fixed_simclr_mask.detach()
            else:
                mask = th.ones(768, device='cuda', requires_grad=False)

            lp_dfs_per_loader = {}
            for loader_name, loader in loaders.items():
                n_dims = 32 * 32 * 3
                df = compute_masked_lpz(model, mask, loader)
                lp_dfs_per_loader[loader_name] = df

            for loader_name, df in lp_dfs_per_loader.items():
                mean_bpd = np.mean(df.bpds)
                result_key_name = f"{loader_name:s}"
                print(f"{result_key_name} BPD: {mean_bpd:.2f}")
                results[f"{result_key_name}_bpd"] = mean_bpd
                writer.add_scalar(f"{result_key_name}_bpd", mean_bpd,
                                  engine.state.epoch)

            for fold in ('train', 'test',):
                itd_df = lp_dfs_per_loader[fold]
                itd_bpds = itd_df.bpds
                itd_diffs = itd_bpds - base_model_bpds[f"{fold}"]
                itd_diffs[~np.isfinite(itd_diffs)] = 300000

                ood_sets = ['ood_test']
                if grey == False:
                    ood_sets.extend(["ood_cifar", "lsun"])
                for ood_name in ood_sets:
                    ood_df = lp_dfs_per_loader[ood_name]
                    metrics = [c for c in ood_df.columns if c != 'bpds']
                    for metric in metrics:
                        auc = compute_auc_for_scores(ood_df[metric], itd_df[metric])

                        results[f"{fold}_vs_{ood_name}_{metric}auc"] = auc
                        print(f"{fold}_vs_{ood_name:10s}_{metric:12s} AUC: {auc:.1%}")
                        writer.add_scalar(f"{fold}_vs_{ood_name}_{metric}_auc", auc,
                                          engine.state.epoch)
                    ood_bpds = ood_df.bpds
                    ood_diffs = ood_bpds - base_model_bpds[ood_name]
                    # set to very high numbers in cas of not finite
                    ood_diffs[~np.isfinite(ood_diffs)] = 300000
                    auc = compute_auc_for_scores(
                        itd_diffs[np.isfinite(itd_diffs)],
                        ood_diffs[np.isfinite(ood_diffs)]) * 100
                    # HACK
                    if shuffle_crop_size is not None:
                        auc = 100 - auc
                    results[f"{fold}_vs_{ood_name}_diff_auc"] = auc
                    print(f"{fold}_vs_{ood_name:10s}_{'diff':12s} AUC: {auc:.1f} %")
                    writer.add_scalar(f"{fold}_vs_{ood_name}_diff_auc", auc,
                                      engine.state.epoch)
                    print()

        engine.state.results = results
        writer.flush()
        if ((engine.state.epoch % max(n_epochs // 5, 1) == 0)
            or engine.state.epoch == n_epochs) and (not debug):
            model_path = os.path.join(output_dir,
                                      f"{engine.state.epoch:d}_model.th")
            th.save(model, open(model_path, 'wb'))

    writer = SummaryWriter(output_dir)
    trainer = Engine(train)
    trainer.add_event_handler(Events.STARTED, evaluate)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate)
    if not debug:
        checkpoint_state_dicts_handler = ModelCheckpoint(output_dir, 'state_dicts',
                                                         save_interval=1,
                                                         n_saved=1,
                                                         require_empty=False,
                                                         save_as_state_dict=True)

        models = dict(model=model)
        if joint_first_scales:
            models['base_model'] = base_model
        if simclr_loss and (simclr_mask_type == 'trained'):
            # necessary as ignite expects 'state_dict' method
            alpha_wrapped = ParameterHolder(nn.Parameter(
                simclr_alphas))
            models['simclr_alphas'] = alpha_wrapped
        optimizers = dict(optimizer=optimizer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  checkpoint_state_dicts_handler,
                                  {**models, **optimizers})
    trainer.run(train_loader, n_epochs)
    return trainer, model

