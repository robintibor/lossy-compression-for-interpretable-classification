import copy
import os
from copy import deepcopy

import higher
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

from invglow.invertible.expression import Expression
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
    # save_image(image_syn_vis, save_name, nrow=ipc)
    # robintibor@gmail.com
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
    ) = get_dataset(dataset, data_path, standardize=False)
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

    # robintibor@gmail.com
    # Load glow
    print("Loading glow...")
    gen = torch.load(
        "/home/schirrmr/data/exps/invertible/pretrain/57/10_model.neurips.th"
    )

    """ organize the real dataset """
    indices_class = [[] for c in range(num_classes)]

    images_all = [
        torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))
    ]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, y in enumerate(labels_all):
        indices_class[y].append(i)
    images_all = torch.cat(images_all, dim=0).to(device)

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

                img_real = image_converter.img_orig_to_clf(img_real)
                lab_real = (
                    torch.ones((img_real.shape[0],), device=device, dtype=torch.long)
                    * c
                )
                this_img_syn_alpha = image_syn_alpha[c * ipc : (c + 1) * ipc].reshape(
                    (ipc, channel, im_size[0], im_size[1])
                )
                img_syn = image_converter.alpha_to_clf(this_img_syn_alpha)
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
            # robintibor@gmail.com: add bpd loss
            # let's standardize! robintibor@gmail.com
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
            # robintibor@gmail.com
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
                )

        loss_avg /= num_classes * outer_loop
        # robintibor@gmail.com
        bpd_avg /= outer_loop

        # robintibor@gmail.com
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
