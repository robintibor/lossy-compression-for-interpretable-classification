import json
import os.path

import kornia
import numpy as np
import torch
import torch as th
from torch import nn

from lossy import data_locations
from lossy import wide_nf_net
from lossy.condensation.networks import ConvNet
from lossy.datasets import get_dataset
from lossy.util import np_to_th


def get_classifier_from_folder(exp_folder, load_weights=True):
    saved_model_folder = exp_folder
    config = json.load(open(
        os.path.join(exp_folder, 'config.json'),
        'r'))
    model_name = config.get("model_name", "wide_nf_net")
    depth = config["depth"]
    if "widen_factor" in config:
        widen_factor = config["widen_factor"]
    else:
        widen_factor = config["width"]
    dropout = config.get("dropout", 0.3)
    optim_type = config.get('optim_type', 'adamw')
    mimic_cxr_target = config.get('mimic_cxr_target', 'pleural_effusion')
    activation = config.get(
        "activation", "relu"
    )  # default was relu
    norm_simple_convnet = config.get(
        "norm_simple_convnet", None
    )
    pooling = config.get("pooling", None)
    external_pretrained_clf = config.get("external_pretrained_clf", False)
    dataset = config['dataset']
    stripes_factor = 0.3

    data_path = data_locations.pytorch_data
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
        batch_size=32,
        standardize=False,
        split_test_off_train=False,
        first_n=None,
        mimic_cxr_target=mimic_cxr_target,
        stripes_factor=stripes_factor,
        eval_batch_size=256,
    )
    mean = wide_nf_net.mean[dataset]
    std = wide_nf_net.std[dataset]

    normalize = kornia.augmentation.Normalize(
        mean=np_to_th(mean, device="cpu", dtype=np.float32),
        std=np_to_th(std, device="cpu", dtype=np.float32),
    )

    # for now just ignore opt_clf stuff
    weight_decay = 1e-5

    lr_clf = 1e-3

    clf, _ = get_clf_and_optim(
            model_name=model_name,
            num_classes=num_classes,
            normalize=normalize,
            optim_type=optim_type,
            saved_model_folder=saved_model_folder if load_weights else None,
            depth=depth,
            widen_factor=widen_factor,
            lr_clf=lr_clf,
            weight_decay=weight_decay,
            activation=activation,
            dataset=dataset,
            norm_simple_convnet=norm_simple_convnet,
            pooling=pooling,
            im_size=im_size,
            external_pretrained_clf=external_pretrained_clf,
    )
    return clf


def get_clf_and_optim(
    model_name,
    num_classes,
    normalize,
    optim_type,
    saved_model_folder,
    depth,
    widen_factor,
    lr_clf,
    weight_decay,
    activation,
    dataset,
    norm_simple_convnet,
    pooling,
    im_size,
    external_pretrained_clf,
):
    if model_name in [
        "wide_nf_net",
        "wide_bnorm_net",
        "ConvNet",
    ]:
        dropout = 0.3
        if saved_model_folder is not None:
            saved_model_config = json.load(
                open(os.path.join(saved_model_folder, "config.json"), "r")
            )
            assert (
                saved_model_config.get("model_name", "wide_nf_net") == model_name
            ), f"{model_name} given but trying to load {saved_model_config['model_name']}"
            depth = saved_model_config["depth"]
            if "widen_factor" in saved_model_config:
                widen_factor = saved_model_config["widen_factor"]
            else:
                widen_factor = saved_model_config["width"]

            dropout = saved_model_config.get("dropout", 0.3)
            activation = saved_model_config.get(
                "activation", "relu"
            )  # default was relu
            norm_simple_convnet = saved_model_config.get(
                "norm_simple_convnet", norm_simple_convnet
            )
            pooling = saved_model_config.get("pooling", pooling)
            assert saved_model_config.get("dataset", "cifar10") == dataset
        if model_name == "wide_nf_net":
            from lossy.wide_nf_net import conv_init, Wide_NFResNet

            nf_net = Wide_NFResNet(
                depth, widen_factor, dropout, num_classes, activation=activation
            ).cuda()
            nf_net.apply(conv_init)
            model = nf_net
        elif model_name == "wide_bnorm_net":
            assert activation == "relu"
            # activation = "relu"  # overwrite for wide resnet for now
            from lossy.wide_resnet import Wide_ResNet, conv_init

            model = Wide_ResNet(
                depth, widen_factor, dropout, num_classes, activation=activation
            ).cuda()
            model.apply(conv_init)
        elif model_name == "ConvNet":
            model = ConvNet(
                3,
                num_classes,
                widen_factor,
                depth,
                activation,
                norm_simple_convnet,
                pooling,
                im_size,
            ).cuda()
        else:
            assert False
    elif model_name == "resnet18":
        from lossy.resnet import resnet18

        model = resnet18(num_classes=num_classes, pretrained=external_pretrained_clf)
    elif model_name == "torchvision_resnet18":
        from torchvision.models import resnet18

        model = resnet18(num_classes=num_classes, pretrained=external_pretrained_clf)
    # What is this? probably old and deletable?
    elif model_name == "conv_net":
        model = ConvNet(
            channel=3,
            num_classes=num_classes,
            net_width=64,
            net_depth=3,
            net_act="shifted_softplus",
            net_norm="batchnorm",
            net_pooling="avgpooling",
            im_size=(32, 32),
        )
    elif model_name == "linear":
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, num_classes),
        )
    else:
        assert False

    clf = nn.Sequential(normalize, model)
    clf = clf.cuda()

    if saved_model_folder is not None:
        assert model_name in [
            "wide_nf_net",
            "wide_bnorm_net",
            "ConvNet",
        ]
        if "lr_preproc" in saved_model_config:
            try:
                # used to have nf_net in name later removed that
                saved_clf_state_dict = th.load(
                    os.path.join(saved_model_folder, "nf_net_state_dict.th")
                )
            except FileNotFoundError:
                saved_clf_state_dict = th.load(
                    os.path.join(saved_model_folder, "clf_state_dict.th")
                )
            # normalization included in model in SimpleBits training
            clf.load_state_dict(saved_clf_state_dict)
        else:
            try:
                # used to have nf_net in name later removed that
                saved_clf_state_dict = th.load(
                    os.path.join(saved_model_folder, "nf_net_state_dict.th")
                )
            except FileNotFoundError:
                saved_clf_state_dict = th.load(
                    os.path.join(saved_model_folder, "state_dict.th")
                )
            # normalization was not included in model in regular training
            clf[1].load_state_dict(saved_clf_state_dict)

    params_with_weight_decay = []
    params_without_weight_decay = []
    for name, param in clf.named_parameters():
        if "weight" in name or "gain" in name:
            params_with_weight_decay.append(param)
        else:
            assert "bias" in name
            params_without_weight_decay.append(param)

    beta_clf = (0.9, 0.995)

    if optim_type == "adam":
        opt_clf = torch.optim.Adam(
            [
                dict(params=params_with_weight_decay, weight_decay=weight_decay),
                dict(params=params_without_weight_decay, weight_decay=0),
            ],
            lr=lr_clf,
            betas=beta_clf,
        )
    else:
        assert optim_type == "adamw"
        opt_clf = torch.optim.AdamW(
            [
                dict(params=params_with_weight_decay, weight_decay=weight_decay),
                dict(params=params_without_weight_decay, weight_decay=0),
            ],
            lr=lr_clf,
            betas=beta_clf,
        )
    return clf, opt_clf
