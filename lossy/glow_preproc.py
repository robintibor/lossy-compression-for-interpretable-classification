from copy import deepcopy
from lossy.image_convert import glow_img_to_img_0_1
from lossy.image_convert import img_0_1_to_glow_img
from torch import nn
from lossy.wide_nf_net import wide_basic


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

