# building on https://github.com/meliketoy/wide-resnet.pytorch
# and https://github.com/vballoli/nfnets-pytorch/search?q=ScaledStdConv2d

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

############### Pytorch CIFAR configuration file ###############
import math
from torch.nn import init
import numpy as np
from lossy.shifted_softplus import ShiftedSoftplus


mean = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408),
    "mnist": (0.1307, 0.1307, 0.1307),
    "fashionmnist": (0.2861, 0.2861, 0.2861),
    "svhn": (0.4377, 0.4438, 0.4728),
}

std = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2675, 0.2565, 0.2761),
    "mnist": (0.3081, 0.3081, 0.3081),
    "fashionmnist": (0.3530, 0.3530, 0.3530),
    "svhn": (0.1980, 0.2010, 0.1970),
}


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)

def learning_rate(init, epoch):
    optim_factor = 0
    if epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1

    return init * math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.
    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    Adapted from timm: https://github.com/rwightman/pytorch-image-models/blob/4ea593196414684d2074cbb81d762f3847738484/timm/models/layers/std_conv.py
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        gain=True,
        gamma=1.0,
        eps=1e-5,
        use_layernorm=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.gain = (
            nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) if gain else None
        )
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        # experimental, slightly faster/less GPU memory use
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(
                self.weight, self.weight.shape[1:], eps=self.eps
            )
        else:
            mean = torch.mean(self.weight, dim=[1, 2, 3], keepdim=True)
            std = torch.std(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(
            x,
            self.get_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


_nonlin_gamma = dict(
    identity=1.0,
    celu=1.270926833152771,
    elu=1.2716004848480225,
    gelu=1.7015043497085571,
    leaky_relu=1.70590341091156,
    log_sigmoid=1.9193484783172607,
    log_softmax=1.0002083778381348,
    relu=1.7139588594436646,
    relu6=1.7131484746932983,
    selu=1.0008515119552612,
    sigmoid=4.803835391998291,
    silu=1.7881293296813965,
    shifted_softplus_1=1.9190,
    shifted_softplus_2=1.8338,
    shifted_softplus_4=1.7630,
    softsign=2.338853120803833,
    softplus=1.9203323125839233,
    tanh=1.5939117670059204,
)


activation_fn = {
    "identity": lambda x, *args, **kwargs: nn.Identity(*args, **kwargs)(x)
    * _nonlin_gamma["identity"],
    "celu": lambda x, *args, **kwargs: nn.CELU(*args, **kwargs)(x)
    * _nonlin_gamma["celu"],
    "elu": lambda x, *args, **kwargs: nn.ELU(*args, **kwargs)(x) * _nonlin_gamma["elu"],
    "gelu": lambda x, *args, **kwargs: nn.GELU(*args, **kwargs)(x)
    * _nonlin_gamma["gelu"],
    "leaky_relu": lambda x, *args, **kwargs: nn.LeakyReLU(*args, **kwargs)(x)
    * _nonlin_gamma["leaky_relu"],
    "log_sigmoid": lambda x, *args, **kwargs: nn.LogSigmoid(*args, **kwargs)(x)
    * _nonlin_gamma["log_sigmoid"],
    "log_softmax": lambda x, *args, **kwargs: nn.LogSoftmax(*args, **kwargs)(x)
    * _nonlin_gamma["log_softmax"],
    "relu": lambda x, *args, **kwargs: nn.ReLU(*args, **kwargs)(x)
    * _nonlin_gamma["relu"],
    "relu6": lambda x, *args, **kwargs: nn.ReLU6(*args, **kwargs)(x)
    * _nonlin_gamma["relu6"],
    "selu": lambda x, *args, **kwargs: nn.SELU(*args, **kwargs)(x)
    * _nonlin_gamma["selu"],
    "sigmoid": lambda x, *args, **kwargs: nn.Sigmoid(*args, **kwargs)(x)
    * _nonlin_gamma["sigmoid"],
    "silu": lambda x, *args, **kwargs: nn.SiLU(*args, **kwargs)(x)
    * _nonlin_gamma["silu"],
    "softplus": lambda x, *args, **kwargs: nn.Softplus(*args, **kwargs)(x)
    * _nonlin_gamma["softplus"],
    "shifted_softplus_1": lambda x, *args, **kwargs: ShiftedSoftplus(1, *args, **kwargs)(x)
    * _nonlin_gamma["shifted_softplus_1"],
    "shifted_softplus_2": lambda x, *args, **kwargs: ShiftedSoftplus(2, *args, **kwargs)(x)
    * _nonlin_gamma["shifted_softplus_2"],
    "shifted_softplus_4": lambda x, *args, **kwargs: ShiftedSoftplus(4, *args, **kwargs)(x)
    * _nonlin_gamma["shifted_softplus_4"],
    "tanh": lambda x, *args, **kwargs: nn.Tanh(*args, **kwargs)(x)
    * _nonlin_gamma["tanh"],
}


def conv3x3(in_planes, out_planes, stride=1):
    return ScaledStdConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class ScalarMultiply(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.scalar * x


class wide_basic(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        dropout_rate,
        stride,
        activation,  # ="relu"
        alpha: float = 0.2,
        beta: float = 1.0,
        zero_init_residual=False,
    ):
        super().__init__()
        conv_class = ScaledStdConv2d
        self.conv1 = conv_class(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv_class(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = conv_class(
                in_planes, planes, kernel_size=1, stride=stride, bias=True
            )
        #self.act = partial(activation_fn[activation], inplace=True)
        self.act = activation_fn[activation]
        self.beta = beta
        self.alpha = alpha

        if zero_init_residual:
            self.final_multiply = ScalarMultiply()
        else:
            self.final_multiply = nn.Identity()

    def forward(self, x):
        # check really before activation as well?
        out = self.act(x) * self.beta
        out = self.dropout(self.conv1(out))
        out = self.conv2(self.act(out))
        out = out * self.alpha
        out = self.final_multiply(out)
        out += self.shortcut(x)

        return out


class Wide_NFResNet(nn.Module):
    def __init__(
        self,
        depth,
        widen_factor,
        dropout_rate,
        num_classes,
        activation,
        verbose=False,
    ):
        super().__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        if verbose:
            print("| Wide-Resnet %dx%d" % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic,
            nStages[1],
            n,
            dropout_rate,
            stride=1,
            activation=activation,
        )
        self.layer2 = self._wide_layer(
            wide_basic,
            nStages[2],
            n,
            dropout_rate,
            stride=2,
            activation=activation,
        )
        self.layer3 = self._wide_layer(
            wide_basic,
            nStages[3],
            n,
            dropout_rate,
            stride=2,
            activation=activation,
        )
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, activation):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(
                    self.in_planes, planes, dropout_rate, stride, activation=activation
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def compute_features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.compute_features(x)
        out = self.linear(out)
        return out
