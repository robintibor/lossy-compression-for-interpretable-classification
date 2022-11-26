from copy import deepcopy
from lossy.image_convert import glow_img_to_img_0_1
from lossy.image_convert import img_0_1_to_glow_img
from torch import nn
import torch as th
import numpy as np
from lossy.wide_nf_net import wide_basic
from lossy.image2image import WrapResidualNonSigmoidUnet
from lossy.image2image import UnetGeneratorCompact
from lossy.affine import AffineOnChans


class OnTopOfGlowPreproc(nn.Module):
    def __init__(self, glow, out_shapes):
        self.glow = [glow] # don't want to show up in parameters
        self.out_shapes = out_shapes
        flatten_modules = []
        for m in glow.modules():
            if m.__class__.__name__ == 'Flatten2d':
                flatten_modules.append(m)
        self.flatten_modules = flatten_modules
        super().__init__()
        self.downsampler_0_1 = nn.Conv2d(out_shapes[0][0], out_shapes[1][0], (2,2), stride=(2,2)).cuda()
        self.downsampler_0_2 = nn.Conv2d(out_shapes[0][0], out_shapes[2][0], (4,4), stride=(4,4)).cuda()
        self.downsampler_1_2 = nn.Conv2d(out_shapes[1][0], out_shapes[2][0], (2,2), stride=(2,2)).cuda()
        unets = []
        for i_o_shape, o_shape in enumerate(out_shapes):
            num_downs = int(np.log2(o_shape[1]).round())
            n_in_chans = o_shape[0]
            n_out_chans = o_shape[0]
            n_in_chans += (n_in_chans * i_o_shape)
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
            two_d_z = [f.invert(a_z)[0] for f,a_z in zip(self.flatten_modules, z)]
            
        out_0 = self.unets[0](two_d_z[0])
        out_1 = self.unets[1](th.cat((two_d_z[1], self.downsampler_0_1(two_d_z[0])), dim=1))
        out_2 = self.unets[2](th.cat((two_d_z[2], self.downsampler_0_2(two_d_z[0]),
                        self.downsampler_1_2(two_d_z[1])), dim=1))

        processed_outs = [f(o)[0] for f, o in zip(self.flatten_modules, (out_0, out_1, out_2))]
        x_inv, _ = self.glow[0].invert(processed_outs)
        x_inv = glow_img_to_img_0_1(x_inv)
        return x_inv


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


## Make into resnet glow encoder instead
class WrapAddLogdet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, fixed=None):
        z = self.module(x)
        return z, 0


def get_glow_preproc(glow, resnet_encoder):
    glow_encoder = deepcopy(glow)
    if resnet_encoder:
        for m in glow_encoder.modules():
            if hasattr(m, 'sequential'):
                if hasattr(m.sequential[0], 'chunk_chans_first'):
                    coef_extractor = m.sequential[1].sequential[2].coef_extractor
                    in_chans = coef_extractor.module[0].conv.in_channels * 2
                    wide_layers = [
                        wide_basic(in_chans, 128, dropout_rate=0, stride=1, activation='elu')]
                    wide_layers.extend(
                        [wide_basic(128, 128, dropout_rate=0, stride=1, activation='elu')
                         for _ in range(5)])
                    wide_layers.append(nn.Conv2d(128, in_chans, (1,1),))

                    m.sequential = nn.Sequential(
                        m.sequential[0],
                        WrapAddLogdet(
                            nn.Sequential(
                                *wide_layers
                            )
                        )
                    )
        glow_encoder.cuda();
    glow_decoder = deepcopy(glow)
    for p in glow_encoder.parameters():
        p.requires_grad_(True)
    for p in glow.parameters():
        p.requires_grad_(False)
    for p in glow_decoder.parameters():
        p.requires_grad_(False)

    preproc = GlowPreproc(glow_encoder, glow_decoder)
    return preproc

