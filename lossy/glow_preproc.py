from copy import deepcopy
from lossy.image_convert import glow_img_to_img_0_1
from lossy.image_convert import img_0_1_to_glow_img
from torch import nn
import torch as th
import numpy as np
from lossy.wide_nf_net import wide_basic
from lossy.image2image import WrapResidualNonSigmoidUnet
from lossy.image2image import WrapResidualAndMixNonSigmoidUnet
from lossy.image2image import UnetGeneratorCompact
from lossy.affine import AffineOnChans
from lossy.invglow.invertible.pure_model import NoLogDet

from lossy.wide_nf_net import wide_basic, conv3x3, ScaledStdConv2d
from lossy.invglow.invertible.view_as import Flatten2d
from lossy.wide_nf_net import wide_basic
from lossy.invglow.invertible.splitter import SubsampleSplitter
from lossy.image2image import CatExtraChannels


def get_glow_preproc(
        glow, encoder_name, cat_clf_chans, merge_weight_clf_chans,):
    glow_encoder = deepcopy(glow)
    # glow_shapes = [(6, 16, 16), (12, 8, 8), (48, 4, 4)]
    if encoder_name == "glow_resnet":
        for m in glow_encoder.modules():
            if hasattr(m, "sequential"):
                if hasattr(m.sequential[0], "chunk_chans_first"):
                    coef_extractor = m.sequential[1].sequential[2].coef_extractor
                    in_chans = coef_extractor.module[0].conv.in_channels * 2
                    if cat_clf_chans:
                        n_chans_to_cat = {12: 64, 24: 128, 48: 128}[in_chans]
                        block_name = {
                            12: "block32x32",
                            24: "block16x16",
                            48: "block8x8",
                        }[in_chans]
                        # n_chans_to_cat *= 4
                        cat_chans_preproc = NoLogDet(
                            SubsampleSplitter(
                                stride=(2, 2),
                                chunk_chans_first=True,
                                checkerboard=False,
                            )
                        )
                        block = CatExtraChannels(
                            wide_basic(
                                in_chans + n_chans_to_cat * 4,
                                128,
                                dropout_rate=0,
                                stride=1,
                                activation="elu",
                            ),
                            n_chans_to_cat,
                            cat_chans_preproc,
                            merge_weight_clf_chans,
                        )
                        wide_layers = [block]
                        setattr(glow_encoder, block_name, block)
                    else:
                        wide_layers = [
                            wide_basic(
                                in_chans,
                                128,
                                dropout_rate=0,
                                stride=1,
                                activation="elu",
                            )
                        ]
                    wide_layers.extend(
                        [
                            wide_basic(
                                128, 128, dropout_rate=0, stride=1, activation="elu"
                            )
                            for _ in range(5)
                        ]
                    )
                    wide_layers.append(
                        nn.Conv2d(
                            128,
                            in_chans,
                            (1, 1),
                        )
                    )
                    m.sequential = nn.Sequential(
                        m.sequential[0], WrapAddLogdet(nn.Sequential(*wide_layers))
                    )
        glow_encoder.cuda()
    elif encoder_name == "resnet":
        glow_encoder = Wide_NFResNet_Encoder(
            16,
            4,
            0,
            "elu",
            cat_clf_chans=cat_clf_chans,
            merge_weight_clf_chans=merge_weight_clf_chans,
        ).cuda()
    else:
        assert encoder_name == "glow"
    glow_decoder = deepcopy(glow)
    for p in glow_encoder.parameters():
        p.requires_grad_(True)
    for p in glow.parameters():
        p.requires_grad_(False)
    for p in glow_decoder.parameters():
        p.requires_grad_(False)

    preproc = GlowPreproc(glow_encoder, glow_decoder)
    return preproc


class GlowPreproc(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = img_0_1_to_glow_img(x)
        z, _ = self.encoder(x)
        # z = [a_z - (a_z - a_z.clamp(-2.75,2.75)).detach() for a_z in z]
        x_inv, _ = self.decoder.invert(z)
        x_inv = glow_img_to_img_0_1(x_inv)
        return x_inv


class FixedLatentPreproc(nn.Module):
    def __init__(self, glow, X):
        # no grad
        self.glow = [glow]
        super().__init__()
        with th.no_grad():
            z, lp = self.glow[0](th.zeros_like(X))
        self.z = nn.ParameterList(
            [nn.Parameter(a_z.clone().detach().requires_grad_(True)) for a_z in z]
        )
        # not really needed but also prevents bug in glow that some things are suddenly assigned as parameters
        self.inds = np.arange(len(X))

    def forward(self, X):
        this_z = [a_z[self.inds] for a_z in self.z]
        simple_X = self.glow[0].invert(this_z)[0] + 0.5

        return simple_X


class Softclamp(nn.Module):
    def __init__(self, a_min, a_max):
        super().__init__()
        self.a_min = a_min
        self.a_max = a_max

    def forward(self, x):
        clamped = th.clamp(x, self.a_min, self.a_max)
        return x + (clamped - x).detach()


class ScaledNonlin(nn.Module):
    def __init__(self, nonlin, scale):
        super().__init__()
        self.nonlin = nonlin
        self.scale = scale

    def forward(self, x):
        return self.scale * self.nonlin(x)


class Wide_NFResNet_Encoder(nn.Module):
    def __init__(
        self,
        depth,
        widen_factor,
        dropout_rate,
        activation,
        cat_clf_chans,
        merge_weight_clf_chans,
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

        self.subsample1 = NoLogDet(
            SubsampleSplitter(stride=(2, 2), chunk_chans_first=True, checkerboard=False)
        )
        self.conv1 = conv3x3(3 * 4, nStages[0] * 4)
        self.in_planes = 16
        self.in_planes = self.in_planes * 4
        if cat_clf_chans:
            self.in_planes += 64 * 4
        self.layer1 = self._wide_layer(
            wide_basic,
            nStages[1],
            n,
            dropout_rate,
            stride=1,
            activation=activation,
        )
        if cat_clf_chans:
            self.layer1 = CatExtraChannels(
                self.layer1, 64, self.subsample1, merge_weight_clf_chans
            )
            self.block32x32 = self.layer1

        self.subsample2 = NoLogDet(
            SubsampleSplitter(stride=(2, 2), chunk_chans_first=True, checkerboard=False)
        )
        self.in_planes = self.in_planes * 4
        if cat_clf_chans:
            self.in_planes += 128 * 4
        self.layer2 = self._wide_layer(
            wide_basic,
            nStages[2],
            n,
            dropout_rate,
            stride=1,
            activation=activation,
        )
        if cat_clf_chans:
            self.layer2 = CatExtraChannels(
                self.layer2, 128, self.subsample2, merge_weight_clf_chans
            )
            self.block16x16 = self.layer2

        self.subsample3 = NoLogDet(
            SubsampleSplitter(stride=(2, 2), chunk_chans_first=True, checkerboard=False)
        )
        self.in_planes = self.in_planes * 4
        if cat_clf_chans:
            self.in_planes += 128 * 4
        self.layer3 = self._wide_layer(
            wide_basic,
            nStages[3],
            n,
            dropout_rate,
            stride=1,
            activation=activation,
        )
        if cat_clf_chans:
            self.layer3 = CatExtraChannels(
                self.layer3, 128, self.subsample3, merge_weight_clf_chans
            )
            self.block8x8 = self.layer3


        self.head1 = nn.Sequential(
            ScaledStdConv2d(
                nStages[1], nStages[1], kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.ELU(),
            ScaledStdConv2d(
                nStages[1], 6, kernel_size=1, stride=1, padding=0, bias=True
            ),
            NoLogDet(Flatten2d()),
        )

        self.head2 = nn.Sequential(
            ScaledStdConv2d(
                nStages[2], nStages[2], kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.ELU(),
            ScaledStdConv2d(
                nStages[2], 12, kernel_size=1, stride=1, padding=0, bias=True
            ),
            NoLogDet(Flatten2d()),
        )

        self.head3 = nn.Sequential(
            ScaledStdConv2d(
                nStages[3], nStages[3], kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.ELU(),
            ScaledStdConv2d(
                nStages[3], 48, kernel_size=1, stride=1, padding=0, bias=True
            ),
            NoLogDet(Flatten2d()),
        )

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
        out = x
        out = self.subsample1(out)
        out = self.conv1(out)
        out = self.layer1(out)
        out1 = self.head1(out)

        out = self.subsample2(out)
        out = self.layer2(out)
        out2 = self.head2(out)

        out = self.subsample3(out)
        out = self.layer3(out)
        out3 = self.head3(out)

        return (
            out1,
            out2,
            out3,
        )

    def forward(self, x):
        out = self.compute_features(x)
        return out, 0  # fakelp


class UnflattenGlow(nn.Module):
    def __init__(self, glow, out_shapes):
        self.glow = [glow]  # don't want to show up in parameters
        self.out_shapes = out_shapes
        flatten_modules = []
        for m in glow.modules():
            if m.__class__.__name__ == "Flatten2d":
                flatten_modules.append(m)
        self.flatten_modules = flatten_modules
        super().__init__()

    def forward(self, x):
        with th.no_grad():
            x = img_0_1_to_glow_img(x)
        z, _ = self.glow[0](x)
        assert len(z) == len(self.flatten_modules)
        two_d_z = [f.invert(a_z)[0] for f, a_z in zip(self.flatten_modules, z)]
        return two_d_z


class OnTopOfGlowMixPreproc(nn.Module):
    def __init__(self, glow, out_shapes):
        self.glow = [glow]  # don't want to show up in parameters
        self.out_shapes = out_shapes
        flatten_modules = []
        for m in glow.modules():
            if m.__class__.__name__ == "Flatten2d":
                flatten_modules.append(m)
        self.flatten_modules = flatten_modules
        super().__init__()
        self.downsampler_0_1 = nn.Conv2d(
            out_shapes[0][0], out_shapes[1][0], (2, 2), stride=(2, 2)
        ).cuda()
        self.downsampler_0_2 = nn.Conv2d(
            out_shapes[0][0], out_shapes[2][0], (4, 4), stride=(4, 4)
        ).cuda()
        self.downsampler_1_2 = nn.Conv2d(
            out_shapes[1][0], out_shapes[2][0], (2, 2), stride=(2, 2)
        ).cuda()
        unets = []
        for i_o_shape, o_shape in enumerate(out_shapes):
            num_downs = int(np.log2(o_shape[1]).round())
            n_in_chans = o_shape[0]
            n_out_chans = o_shape[0]
            # add chans from downsampling part
            n_in_chans += n_in_chans * i_o_shape
            this_unet = WrapResidualAndMixNonSigmoidUnet(
                UnetGeneratorCompact(
                    n_in_chans,
                    n_out_chans * 2,
                    num_downs=num_downs,
                    final_nonlin=nn.Identity,
                    norm_layer=AffineOnChans,
                    nonlin_down=nn.ELU,
                    nonlin_up=nn.ELU,
                ),
            )
            unets.append(this_unet)
        self.unets = nn.ModuleList(unets)

    def forward(self, x):
        with th.no_grad():
            x = img_0_1_to_glow_img(x)
            z, _ = self.glow[0](x)
            assert len(z) == len(self.flatten_modules)
            two_d_z = [f.invert(a_z)[0] for f, a_z in zip(self.flatten_modules, z)]

        out_0 = self.unets[0](two_d_z[0])
        out_1 = self.unets[1](
            th.cat((two_d_z[1], self.downsampler_0_1(two_d_z[0])), dim=1)
        )
        out_2 = self.unets[2](
            th.cat(
                (
                    two_d_z[2],
                    self.downsampler_0_2(two_d_z[0]),
                    self.downsampler_1_2(two_d_z[1]),
                ),
                dim=1,
            )
        )

        processed_outs = [
            f(o)[0] for f, o in zip(self.flatten_modules, (out_0, out_1, out_2))
        ]
        x_inv, _ = self.glow[0].invert(processed_outs)
        x_inv = glow_img_to_img_0_1(x_inv)
        return x_inv


class OnTopOfGlowPreproc(nn.Module):
    def __init__(self, glow, out_shapes):
        self.glow = [glow]  # don't want to show up in parameters
        self.out_shapes = out_shapes
        flatten_modules = []
        for m in glow.modules():
            if m.__class__.__name__ == "Flatten2d":
                flatten_modules.append(m)
        self.flatten_modules = flatten_modules
        super().__init__()
        self.downsampler_0_1 = nn.Conv2d(
            out_shapes[0][0], out_shapes[1][0], (2, 2), stride=(2, 2)
        ).cuda()
        self.downsampler_0_2 = nn.Conv2d(
            out_shapes[0][0], out_shapes[2][0], (4, 4), stride=(4, 4)
        ).cuda()
        self.downsampler_1_2 = nn.Conv2d(
            out_shapes[1][0], out_shapes[2][0], (2, 2), stride=(2, 2)
        ).cuda()
        unets = []
        for i_o_shape, o_shape in enumerate(out_shapes):
            num_downs = int(np.log2(o_shape[1]).round())
            n_in_chans = o_shape[0]
            n_out_chans = o_shape[0]
            # add chans from downsampling part
            n_in_chans += n_in_chans * i_o_shape
            this_unet = WrapResidualNonSigmoidUnet(
                UnetGeneratorCompact(
                    n_in_chans,
                    n_out_chans,
                    num_downs=num_downs,
                    final_nonlin=nn.Identity,
                    norm_layer=AffineOnChans,
                    nonlin_down=nn.ELU,
                    nonlin_up=nn.ELU,
                ),
            )
            unets.append(this_unet)
        self.unets = nn.ModuleList(unets)

    def forward(self, x):
        with th.no_grad():
            x = img_0_1_to_glow_img(x)
            z, _ = self.glow[0](x)
            assert len(z) == len(self.flatten_modules)
            two_d_z = [f.invert(a_z)[0] for f, a_z in zip(self.flatten_modules, z)]

        out_0 = self.unets[0](two_d_z[0])
        out_1 = self.unets[1](
            th.cat((two_d_z[1], self.downsampler_0_1(two_d_z[0])), dim=1)
        )
        out_2 = self.unets[2](
            th.cat(
                (
                    two_d_z[2],
                    self.downsampler_0_2(two_d_z[0]),
                    self.downsampler_1_2(two_d_z[1]),
                ),
                dim=1,
            )
        )

        processed_outs = [
            f(o)[0] for f, o in zip(self.flatten_modules, (out_0, out_1, out_2))
        ]
        x_inv, _ = self.glow[0].invert(processed_outs)
        x_inv = glow_img_to_img_0_1(x_inv)
        return x_inv


## Make into resnet glow encoder instead
class WrapAddLogdet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, fixed=None):
        z = self.module(x)
        return z, 0
