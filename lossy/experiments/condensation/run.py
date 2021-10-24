import copy
import os
import sys
from copy import deepcopy

import higher
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

from lossy.glow import load_normal_glow
from lossy.invglow.invertible.expression import Expression
from lossy.augment import TrivialAugmentPerImage
from lossy.image_convert import (
    ImageConverter,
    glow_img_to_img_0_1,
    img_0_1_to_cifar100_standardized,
)
from lossy.condensation.utils import (
    get_loops,
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset,
    get_daparam,
    match_loss,
    get_time,
    TensorDataset,
    epoch,
)


def save_image_from_alpha(
    image_filename, image_syn_alpha, image_converter, num_classes
):
    """ visualize and save """
    image_syn_vis = copy.deepcopy(
        image_converter.alpha_to_img_orig(image_syn_alpha).detach().cpu()
    )
    image_syn_vis[image_syn_vis < 0] = 0.0
    image_syn_vis[image_syn_vis > 1] = 1.0
    save_image(image_syn_vis, image_filename, nrow=num_classes)
    # The generated images would be slightly different from the visualization results in the paper,
    # because of the initialization and normalization of pixels.


def evaluate_models(
    image_syn_alpha,
    label_syn,
    dataset,
    ipc,
    testloader,
    lr_net,
    batch_train,
    num_eval,
    model_eval_pool,
    channel,
    model_name,
    im_size,
    net_norm,
    net_act,
    num_classes,
    image_converter,
    epoch_eval_train,
    saved_model,
    trivial_augment,
    same_aug_across_batch,
):
    accs_per_model = {}
    for key in model_eval_pool:
        accs_per_model[key] = []
    for model_eval_name in model_eval_pool:
        print("\nEvaluation\nmodel_eval = %s" % model_eval_name)
        param_augment = get_daparam(dataset, model_name, model_eval_name, ipc)
        if param_augment["strategy"] != "none":
            # More epochs for evaluation with augmentation will be better.
            epoch_eval_train = 1000
            print("data augmentation = %s" % param_augment)
        else:
            epoch_eval_train = epoch_eval_train
        accs = []
        for it_eval in range(num_eval):
            net_eval = get_network(
                model_eval_name,
                channel,
                num_classes,
                im_size,
                net_norm_override=net_norm,
                net_act_override=net_act,
            ).to(
                image_syn_alpha.device
            )  # get a random model
            # Override net with saved model if given
            if saved_model is not None:
                net_eval = copy_with_replaced_head(saved_model, num_classes)
            # will unnecessary add noise I guess?
            image_syn_eval = copy.deepcopy(
                image_converter.alpha_to_img_orig(image_syn_alpha).detach()
            )
            label_syn_eval = copy.deepcopy(
                label_syn.detach()
            )  # avoid any unaware modification
            _, acc_train, acc_test = evaluate_synset(
                it_eval,
                net_eval,
                image_syn_eval,
                label_syn_eval,
                testloader,
                lr_net,
                batch_train,
                param_augment,
                image_syn_alpha.device,
                epoch_eval_train,
                image_converter=image_converter,
                trivial_augment=trivial_augment,
                same_aug_across_batch=same_aug_across_batch,
            )
            accs.append(acc_test)
        print(
            "Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------"
            % (len(accs), model_eval_name, np.mean(accs), np.std(accs))
        )
        accs_per_model[model_eval_name] = accs
    return accs_per_model


def copy_with_replaced_head(saved_model, num_classes):
    net = copy.deepcopy(saved_model)
    if hasattr(net, "classifier"):
        net.classifier = nn.Linear(2048, num_classes, bias=True).cuda()
    else:
        assert hasattr(net[2], "fc")
        net[2].fc = nn.Linear(64, num_classes, bias=True).cuda()
    return net


def th_to_np(x):
    return x.detach().cpu().numpy()


def get_pretrained_net(
    pretrain_dataset,
    data_path,
    mimic_cxr_clip,
    mimic_cxr_target,
    model_name,
    net_norm,
    net_act,
    lr_net,
    image_converter,
    n_epochs,
):
    tqdm = lambda x: x
    trange = range
    import torch as th

    device = "cuda"
    (
        channel,
        im_size,
        num_pretrain_classes,
        _,
        _,
        _,
        dst_pretrain_train,
        dst_pretrain_test,
        _,
    ) = get_dataset(
        pretrain_dataset,
        data_path,
        standardize=False,
        mimic_cxr_clip=mimic_cxr_clip,
        mimic_cxr_target=mimic_cxr_target,
    )
    # strings with model names for evaluation models

    pretrained_net = get_network(
        model_name,
        channel,
        num_pretrain_classes,
        im_size,
        net_norm_override=net_norm,
        net_act_override=net_act,
    ).to(device)

    optimizer_pretrain = torch.optim.SGD(
        pretrained_net.parameters(), lr=lr_net, momentum=0.9, weight_decay=0.0005
    )

    pretrainloader = th.utils.data.DataLoader(
        dst_pretrain_train, shuffle=True, drop_last=True, batch_size=128, num_workers=2
    )

    lr_schedule = [n_epochs // 2 + 1]
    cur_lr_net = lr_net
    criterion = nn.CrossEntropyLoss()
    for i_epoch in trange(n_epochs):
        for X, y in pretrainloader:
            X = X.cuda()
            y = y.cuda()
            img = image_converter.img_orig_to_clf(X)
            ## augment?
            output = pretrained_net(img)
            loss = criterion(output, y)
            optimizer_pretrain.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_pretrain.step()
            optimizer_pretrain.zero_grad(set_to_none=True)
        if i_epoch in lr_schedule:
            cur_lr_net *= 0.1
            optimizer_pretrain = torch.optim.SGD(
                pretrained_net.parameters(),
                lr=cur_lr_net,
                momentum=0.9,
                weight_decay=0.0005,
            )
    pretrain_testloader = th.utils.data.DataLoader(
        dst_pretrain_test, shuffle=False, drop_last=False, batch_size=128, num_workers=2
    )
    import pandas as pd

    eval_df = pd.DataFrame()
    for X, y in tqdm(pretrain_testloader):
        X = X.cuda()
        y = y.cuda()
        with th.no_grad():
            img = image_converter.img_orig_to_clf(X)
            pred_X = th.softmax(pretrained_net(img), dim=1)
            correct_prob = th.stack([p[a_y] for p, a_y in zip(pred_X, y)])
        eval_df = pd.concat(
            (
                eval_df,
                pd.DataFrame(
                    dict(
                        correct_prob=th_to_np(correct_prob),
                        correct=th_to_np(pred_X).argmax(axis=1) == th_to_np(y),
                        y=th_to_np(y),
                    )
                ),
            ),
            ignore_index=True,
        )
    print(eval_df.correct.mean())
    return pretrained_net


def run_exp(
    data_path,
    dataset,
    ipc,
    n_outer_epochs,
    model_name,
    batch_train,
    batch_real,
    init,
    lr_img,
    lr_net,
    dis_metric,
    epoch_eval_train,
    num_exp,
    num_eval,
    eval_mode,
    save_path,
    bpd_loss_weight,
    image_standardize_before_glow,
    img_alpha_init_factor,
    sigmoid_on_alpha,
    standardize_for_clf,
    net_norm,
    net_act,
    loss_name,
    optim_class_img,
    outer_loop,
    inner_loop,
    saved_model_path,
    rescale_grads,
    glow_noise_on_out,
    trivial_augment,
    same_aug_across_batch,
    mimic_cxr_clip,
    mimic_cxr_target,
    pretrain_dataset,
):
    # outer_loop, inner_loop = get_loops(ipc)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # The list of iterations when we evaluate models and record results.
    i_eval_epochs = (
        np.arange(0, n_outer_epochs + 1, min(500, n_outer_epochs)).tolist()
        if eval_mode == "S"
        else [n_outer_epochs]
    )
    print("Evaluations at iterations:", i_eval_epochs)
    (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
    ) = get_dataset(
        dataset,
        data_path,
        standardize=False,
        mimic_cxr_clip=mimic_cxr_clip,
        mimic_cxr_target=mimic_cxr_target,
    )
    # strings with model names for evaluation models
    model_eval_pool = get_eval_pool(eval_mode, model_name, model_name)

    image_converter = ImageConverter(
        image_standardize_before_glow=image_standardize_before_glow,
        sigmoid_on_alpha=sigmoid_on_alpha,
        standardize_for_clf=standardize_for_clf,
        glow_noise_on_out=glow_noise_on_out,
    )

    data_save = []

    total_loss_weight = 10
    print("Evaluation model pool: ", model_eval_pool)

    # Load glow
    print("Loading glow...")
    gen = load_normal_glow()

    """ organize the real dataset """

    train_det_loader = torch.utils.data.DataLoader(
        dst_train,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    Xs_ys = [(X, y) for X, y in train_det_loader]

    images_all = torch.cat([X for X, y in Xs_ys]).to(device)
    labels_all = torch.cat([y for X, y in Xs_ys])

    indices_class = [[] for c in range(num_classes)]
    for i, y in enumerate(labels_all):
        indices_class[y].append(i)

    for c in range(num_classes):
        print("class c = %d: %d real images" % (c, len(indices_class[c])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    """ initialize the synthetic data """
    image_syn_alpha = torch.randn(
        size=(num_classes * ipc, channel, im_size[0], im_size[1]),
        dtype=torch.float,
        requires_grad=True,
        device=device,
    )
    label_syn = torch.tensor(
        [np.ones(ipc) * i for i in range(num_classes)],
        dtype=torch.long,
        requires_grad=False,
        device=device,
    ).view(
        -1
    )  # [0,0,0, 1,1,1, ..., 9,9,9]

    print("initialize synthetic data from random noise")
    image_syn_alpha.data[:] = image_syn_alpha.data[:] * img_alpha_init_factor

    # Initialize Optimizer/Loss
    if optim_class_img == "sgd":
        optimizer_img = torch.optim.SGD(
            [
                image_syn_alpha,
            ],
            lr=lr_img,
            momentum=0.5,
        )  # optimizer_img for synthetic data
    else:
        assert optim_class_img == "adam"
        optimizer_img = torch.optim.Adam(
            [
                image_syn_alpha,
            ],
            lr=lr_img,
            betas=(0.5, 0.99),
            eps=1e-6,
        )  # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    criterion = nn.CrossEntropyLoss()

    print("%s training begins" % get_time())

    grad_init_std = None
    if saved_model_path is not None:
        print("Loading saved model...")
        if "#" in saved_model_path:
            repo, model_variant_name = saved_model_path.split("#")
            saved_model = torch.hub.load(repo, model_variant_name, pretrained=True)
            saved_model = saved_model.cuda()
            saved_model = nn.Sequential(
                Expression(glow_img_to_img_0_1),
                Expression(img_0_1_to_cifar100_standardized),
                saved_model,
            ).cuda()
        else:
            saved_model = torch.load(saved_model_path)
    else:
        saved_model = None

    if pretrain_dataset is not None:
        n_epochs_pretrain = 50
        pretrained_net = get_pretrained_net(
            pretrain_dataset,
            data_path,
            mimic_cxr_clip,
            mimic_cxr_target,
            model_name,
            net_norm,
            net_act,
            lr_net,
            image_converter,
            n_epochs_pretrain,
        )
    else:
        pretrained_net = None

    for i_outer_epoch in range(n_outer_epochs + 1):
        """ Evaluate synthetic data """
        if i_outer_epoch in i_eval_epochs:
            accs_per_model = evaluate_models(
                image_syn_alpha,
                label_syn,
                dataset,
                ipc,
                testloader,
                lr_net,
                batch_train,
                num_eval,
                model_eval_pool,
                channel,
                model_name,
                im_size,
                net_norm,
                net_act,
                num_classes,
                image_converter,
                epoch_eval_train,
                saved_model,
                trivial_augment=trivial_augment,
                same_aug_across_batch=False,  # works better with false than passing argument
            )
            image_filename = os.path.join(
                save_path,
                "vis_%s_%s_%dipc_iter%d.png"
                % (dataset, model_name, ipc, i_outer_epoch),
            )
            save_image_from_alpha(
                image_filename, image_syn_alpha, image_converter, num_classes
            )

        """ Train synthetic data """
        net = get_network(
            model_name,
            channel,
            num_classes,
            im_size,
            net_norm_override=net_norm,
            net_act_override=net_act,
        ).to(
            device
        )  # get a random model
        # Override net with saved model if given
        if saved_model is not None:
            net = copy_with_replaced_head(saved_model, num_classes)
        # Override net with pretrained net (but don't use for evaluation)
        if pretrained_net is not None:
            net = copy_with_replaced_head(pretrained_net, num_classes)
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(
            net.parameters(), lr=lr_net, momentum=0.5
        )  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        loss_avg = 0
        bpd_avg = 0

        for ol in range(outer_loop):
            """ freeze the running mu and sigma for BatchNorm layers """
            # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
            # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
            # This would make the model with BatchNorm layers easier to train.
            BN_flag = False
            BNSizePC = 16  # for batch normalization
            for module in net.modules():
                if "BatchNorm" in module._get_name():  # BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat(
                    [get_images(c, BNSizePC) for c in range(num_classes)], dim=0
                )
                net.train()  # for updating the mu, sigma of BatchNorm
                img_real = image_converter.img_orig_to_clf(img_real)

                _ = net(img_real)  # get running mu, sigma
                for module in net.modules():
                    if "BatchNorm" in module._get_name():  # BatchNorm
                        module.eval()  # fix mu and sigma of every BatchNorm layer

            """ update synthetic data """
            sum_loss_over_classes = torch.tensor(0.0).to(device)
            optimizer_img.zero_grad()
            weight = total_loss_weight / (1 + bpd_loss_weight) * num_classes
            all_gw_reals = []
            for c in range(num_classes):
                img_real = get_images(c, batch_real)
                if trivial_augment and same_aug_across_batch:
                    aug_m = TrivialAugmentPerImage(
                        1,
                        num_magnitude_bins=31,
                        std_aug_magnitude=None,
                        extra_augs=True,
                        same_across_batch=same_aug_across_batch,
                    )
                    img_real = aug_m(img_real)
                img_real = image_converter.img_orig_to_clf(img_real)
                lab_real = (
                    torch.ones((img_real.shape[0],), device=device, dtype=torch.long)
                    * c
                )
                this_img_syn_alpha = image_syn_alpha[c * ipc : (c + 1) * ipc].reshape(
                    (ipc, channel, im_size[0], im_size[1])
                )

                img_syn_orig = image_converter.alpha_to_img_orig(this_img_syn_alpha)
                if trivial_augment and (not same_aug_across_batch):
                    aug_m = TrivialAugmentPerImage(
                        img_syn_orig.shape[0],
                        num_magnitude_bins=31,
                        std_aug_magnitude=None,
                        extra_augs=True,
                        same_across_batch=same_aug_across_batch,
                    )
                if trivial_augment:
                    img_syn_orig = aug_m(img_syn_orig)
                img_syn = image_converter.img_orig_to_clf(img_syn_orig)
                lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
                if loss_name != "higher":
                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(
                        loss_syn, net_parameters, create_graph=True
                    )

                    if loss_name == "match_loss":
                        loss = match_loss(gw_syn, gw_real, dis_metric, device)
                        weighted_loss = weight * loss
                        weighted_loss.backward()
                        sum_loss_over_classes += loss.item()
                    elif loss_name == "grad_grad":

                        # fixed multiplying with 200 to achieve similar-value-range grads
                        # as with matching loss
                        torch.autograd.backward(
                            gw_syn, grad_tensors=[-weight * g for g in gw_real]
                        )
                        sum_loss_over_classes += 0
                    else:
                        assert loss_name == "aggregated_grad_grad"
                        all_gw_reals.append(gw_real)
                else:
                    assert loss_name == "higher"
                    with higher.innerloop_ctx(
                        net, optimizer_net, copy_initial_weights=True
                    ) as (f_net, f_opt):
                        output_syn = f_net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn)
                        f_opt.step(loss_syn)
                        output_real = f_net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        loss_real.backward()
                    sum_loss_over_classes += loss_real.item()
            if loss_name == "aggregated_grad_grad":
                img_syn = image_converter.alpha_to_clf(image_syn_alpha)
                lab_syn = torch.repeat_interleave(torch.arange(num_classes), ipc).to(
                    img_syn.device
                )
                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(
                    loss_syn, net_parameters, create_graph=True
                )
                with torch.no_grad():
                    gw_real_sum = [
                        torch.sum(torch.stack([g[i_w] for g in all_gw_reals]), dim=0)
                        for i_w in range(len(all_gw_reals[0]))
                    ]
                # fixed multiplying with 200 to achieve similar-value-range grads
                # as with matching loss
                torch.autograd.backward(
                    gw_syn, grad_tensors=[-weight * g for g in gw_real_sum]
                )
            if rescale_grads and (
                loss_name in ["higher", "grad_grad", "aggregated_grad_grad"]
            ):
                if grad_init_std is None:
                    grad_init_std = image_syn_alpha.grad.std().item()
                image_syn_alpha.grad.data[:] = (
                    image_syn_alpha.grad.data[:] * weight / (grad_init_std * 10)
                )
            else:
                assert (not rescale_grads) or (loss_name == "match_loss")
            bpd_loss_applied = (bpd_loss_weight is not None) and (bpd_loss_weight > 0)
            with torch.set_grad_enabled(bpd_loss_applied):
                all_bpds = []
                n_chunks = len(
                    torch.chunk(image_syn_alpha, max(len(image_syn_alpha) // 50, 1), 0)
                )
                for i_chunk in range(n_chunks):
                    this_image_syn_alpha = torch.chunk(
                        image_syn_alpha, max(len(image_syn_alpha) // 50, 1), 0
                    )[i_chunk]
                    img_for_glow = image_converter.alpha_to_glow(this_image_syn_alpha)
                    bpd = -(
                        gen(img_for_glow)[1]
                        - np.log(256) * np.prod(img_for_glow.shape[1:])
                    ) / (np.log(2) * np.prod(img_for_glow.shape[1:]))
                    adjusted_bpd_loss_weight = (2 * bpd_loss_weight) / (
                        bpd_loss_weight + 1
                    )
                    bpd_loss = adjusted_bpd_loss_weight * torch.mean(bpd)
                    if bpd_loss_applied and (torch.all(torch.isfinite(bpd)).item()):
                        bpd_loss.backward()
                    all_bpds.append(bpd.detach().cpu().numpy())
                    if not bpd_loss_applied:
                        del bpd_loss, bpd, img_for_glow, this_image_syn_alpha
                all_bpds = np.concatenate(all_bpds)

            optimizer_img.step()
            loss_avg += sum_loss_over_classes
            bpd_avg += all_bpds.mean()
            if bpd_loss_applied:
                del bpd_loss, bpd, img_for_glow

            if ol == outer_loop - 1:
                break

            """ update network to be unified with optimizer_img xxxxxxxxxxxxxxxxx """
            # avoid any unaware modification
            image_syn_train, label_syn_train = (
                copy.deepcopy(
                    image_converter.alpha_to_img_orig(image_syn_alpha).detach()
                ),
                copy.deepcopy(label_syn.detach()),
            )
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(
                dst_syn_train,
                batch_size=256,
                shuffle=True,
                num_workers=0,
            )
            for il in range(inner_loop):
                epoch(
                    "train",
                    trainloader,
                    net,
                    optimizer_net,
                    criterion,
                    None,
                    device,
                    image_converter,
                    trivial_augment=trivial_augment,
                    same_aug_across_batch=False,
                )

        loss_avg /= num_classes * outer_loop
        bpd_avg /= outer_loop

        if i_outer_epoch % 10 == 0:
            print(
                "%s iter = %04d, loss = %.4f bpd = %.4f"
                % (get_time(), i_outer_epoch, loss_avg, bpd_avg)
            )

        if i_outer_epoch == n_outer_epochs:  # only record the final results
            data_save.append(
                [
                    copy.deepcopy(image_syn_alpha.detach().cpu()),
                    copy.deepcopy(label_syn.detach().cpu()),
                ]
            )
            torch.save(
                {
                    "data": data_save,
                    "accs_per_model": accs_per_model,
                },
                os.path.join(
                    save_path, "res_%s_%s_%dipc.pt" % (dataset, model_name, ipc)
                ),
            )
            results = {"accs": accs_per_model, "bpd": bpd_avg}

    print("\n==================== Final Results ====================\n")
    for key in model_eval_pool:
        accs = accs_per_model[key]
        print(
            "Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%"
            % (
                num_exp,
                model_name,
                len(accs),
                key,
                np.mean(accs) * 100,
                np.std(accs) * 100,
            )
        )

    return results

if __name__ == '__main__':
    from lossy import data_locations
    pretrain_dataset = None
    glow_noise_on_out = True
    n_outer_epochs = 1000
    bpd_loss_weight = 10
    img_alpha_init_factor = 0.2
    image_standardize_before_glow = False
    sigmoid_on_alpha = True
    lr_img = 0.1
    lr_net = 1e-2
    standardize_for_clf = False
    net_norm = "instancenorm"
    net_act = "relu"
    optim_class_img = "adam"
    loss_name = "match_loss"
    rescale_grads = False
    saved_model_path = None
    trivial_augment = False
    same_aug_across_batch = False
    mimic_cxr_target = None  # ['gender', 'age', 'disease']
    ipc = 1
    outer_loop = 1
    inner_loop = 1
    dataset = 'CIFAR10'
    mimic_cxr_clip = 1.0
    data_path = data_locations.pytorch_data
    model_name = "ConvNet"
    batch_real = 256
    batch_train = 256
    init = "noise"
    dis_metric = "ours"
    epoch_eval_train = 300
    num_exp = 1
    num_eval = 5
    eval_mode = "S"
    save_path = "."

    run_exp(
        data_path,
        dataset,
        ipc,
        n_outer_epochs,
        model_name,
        batch_train,
        batch_real,
        init,
        lr_img,
        lr_net,
        dis_metric,
        epoch_eval_train,
        num_exp,
        num_eval,
        eval_mode,
        save_path,
        bpd_loss_weight,
        image_standardize_before_glow,
        img_alpha_init_factor,
        sigmoid_on_alpha,
        standardize_for_clf,
        net_norm,
        net_act,
        loss_name,
        optim_class_img,
        outer_loop,
        inner_loop,
        saved_model_path,
        rescale_grads,
        glow_noise_on_out,
        trivial_augment,
        same_aug_across_batch,
        mimic_cxr_clip,
        mimic_cxr_target,
        pretrain_dataset,
    )
