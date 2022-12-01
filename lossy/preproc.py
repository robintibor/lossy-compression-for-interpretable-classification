import json
import os.path
from functools import partial

import torch as th
from torch import nn

from lossy.affine import AffineOnChans
from lossy.glow import load_glow
from lossy.glow_preproc import FixedLatentPreproc
from lossy.glow_preproc import OnTopOfGlowMixPreproc
from lossy.glow_preproc import OnTopOfGlowPreproc
from lossy.glow_preproc import UnflattenGlow
from lossy.glow_preproc import get_glow_preproc
from lossy.image2image import UnetGeneratorWithExtraInput
from lossy.image2image import WrapResidualAndBlendUnet, WrapResidualAndMixUnet
from lossy.image2image import WrapResidualAndMixGreyUnet
from lossy.image2image import WrapResidualIdentityUnet, UnetGenerator
from lossy.image_convert import add_glow_noise_to_0_1
from lossy.image_convert import quantize_data
from lossy.image_convert import soft_clamp_to_0_1
from lossy.invglow.invertible.affine import AffineModifierClampEps
from lossy.modules import Expression


def get_preprocessor(preproc_name, glow, encoder_clip_eps, cat_clf_chans_for_preproc,
                     merge_weight_clf_chans, unet_use_bias, quantize_after_simplifier,
                     noise_after_simplifier, soft_clamp_0_1,
                     X
                     ):
    def to_plus_minus_one(x):
        return (x * 2) - 1

    if preproc_name == "unet":
        preproc = nn.Sequential(
            Expression(to_plus_minus_one),
            UnetGenerator(
                3,
                3,
                num_downs=5,
                final_nonlin=nn.Sigmoid,
                norm_layer=AffineOnChans,
                nonlin_down=nn.ELU,
                nonlin_up=nn.ELU,
                use_bias=unet_use_bias,
            ),
        ).cuda()
    elif preproc_name == "res_unet":
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
                    use_bias=unet_use_bias,
                ),
            ),
            final_nonlin=nn.Sigmoid(),
        ).cuda()
    elif preproc_name == "res_blend_unet":
        preproc = WrapResidualAndBlendUnet(
            nn.Sequential(
                Expression(to_plus_minus_one),
                UnetGenerator(
                    3,
                    6,
                    num_downs=5,
                    final_nonlin=nn.Identity,
                    norm_layer=AffineOnChans,
                    nonlin_down=nn.ELU,
                    nonlin_up=nn.ELU,
                    use_bias=unet_use_bias,
                ),
            ),
        ).cuda()
    elif preproc_name == "res_mix_unet":
        preproc = WrapResidualAndMixUnet(
            nn.Sequential(
                Expression(to_plus_minus_one),
                UnetGenerator(
                    3,
                    6,
                    num_downs=5,
                    final_nonlin=nn.Identity,
                    norm_layer=AffineOnChans,
                    nonlin_down=nn.ELU,
                    nonlin_up=nn.ELU,
                    use_bias=unet_use_bias,
                ),
            ),
        ).cuda()
    elif preproc_name == "res_mix_grey_unet":
        preproc = WrapResidualAndMixGreyUnet(
            nn.Sequential(
                Expression(to_plus_minus_one),
                UnetGenerator(
                    3,
                    7,
                    num_downs=5,
                    ngf=64,  # 64
                    final_nonlin=nn.Identity,
                    norm_layer=AffineOnChans,  # SimpleLayerNorm,#nn.BatchNorm2d,#AffineOnChans,
                    nonlin_down=nn.ELU,
                    nonlin_up=nn.ELU,
                    use_bias=unet_use_bias,
                ),
            ),
        ).cuda()
    elif preproc_name == "res_mix_glow_unet":
        glow_out_shapes = [(6, 16, 16), (12, 8, 8), (48, 4, 4)]
        unflat_glow = UnflattenGlow(glow, glow_out_shapes)
        preproc = WrapResidualAndMixUnet(
            nn.Sequential(
                Expression(to_plus_minus_one),
                UnetGeneratorWithExtraInput(
                    3,
                    6,
                    num_downs=5,
                    final_nonlin=partial(AffineOnChans, 6),  # nn.Identity,
                    norm_layer=AffineOnChans,
                    nonlin_down=nn.ELU,
                    nonlin_up=nn.ELU,
                    use_bias=unet_use_bias,
                ),
            ),
        ).cuda()
        preproc.unet[1].model.model[4].factors.data[:] = 0.2
    elif preproc_name == "glow":
        preproc = get_glow_preproc(
            glow,
            encoder_name="glow",
            cat_clf_chans=cat_clf_chans_for_preproc,
            merge_weight_clf_chans=None,
        )
    elif preproc_name == "on_top_of_glow":
        out_shapes = [(6, 16, 16), (12, 8, 8), (48, 4, 4)]
        preproc = OnTopOfGlowPreproc(glow, out_shapes).cuda()
    elif preproc_name == "on_top_of_glow_mix":
        out_shapes = [(6, 16, 16), (12, 8, 8), (48, 4, 4)]
        preproc = OnTopOfGlowMixPreproc(glow, out_shapes).cuda()
    elif preproc_name == "latent_z":
        # first test batch is batch you mostly used for debugging
        preproc = FixedLatentPreproc(glow, X)
    elif preproc_name == "glow_with_pure_resnet":
        preproc = get_glow_preproc(
            glow=glow,
            encoder_name="resnet",
            cat_clf_chans=cat_clf_chans_for_preproc,
            merge_weight_clf_chans=merge_weight_clf_chans,
        )
    elif preproc_name == "res_glow_with_pure_resnet":
        preproc = get_glow_preproc(
            glow=glow,
            encoder_name="res_glow_resnet",
            cat_clf_chans=cat_clf_chans_for_preproc,
            merge_weight_clf_chans=merge_weight_clf_chans,
        )
    else:
        assert preproc_name == "glow_with_resnet"
        preproc = get_glow_preproc(
            glow,
            encoder_name="glow_resnet",
            cat_clf_chans=cat_clf_chans_for_preproc,
            merge_weight_clf_chans=merge_weight_clf_chans,
        )

    if preproc_name in [
        "glow_with_resnet",
        "glow_with_pure_resnet",
    ]:
        for m in preproc.decoder.modules():
            if hasattr(m, 'modifier'):
                m.modifier = AffineModifierClampEps(
                    m.modifier.sigmoid_or_exp_scale,
                    m.modifier.add_first,
                    encoder_clip_eps)

    if preproc_name in [
        "glow",
        "on_top_of_glow",
        "on_top_of_glow_mix",
    ]:  # Old style, not sure if good
        for m in glow.modules():
            if m.__class__.__name__ == "AffineModifier":
                m.eps = 5e-2

    preproc_post = nn.Sequential()
    if quantize_after_simplifier:
        preproc_post.add_module("quantize", Expression(quantize_data))
    if noise_after_simplifier:
        preproc_post.add_module("add_glow_noise", Expression(add_glow_noise_to_0_1))
    if soft_clamp_0_1:
        preproc_post.add_module("soft_clamp_to_0_1", Expression(soft_clamp_to_0_1))
    preproc = nn.Sequential(preproc, preproc_post)
    preproc.eval()
    return preproc


def get_preprocessor_from_folder(saved_exp_folder, X=None, glow="reload"):
    config = json.load(open(os.path.join(saved_exp_folder, "config.json"), "r"))
    if 'preproc_name' in config:
        preproc_name = config['preproc_name']
    else:
        preproc_name = ["unet", "res_unet"][config.get("residual_preproc", True)]
    unet_use_bias = config.get("unet_use_bias", False)
    cat_clf_chans_for_preproc  = config.get("cat_clf_chans_for_preproc", False)
    merge_weight_clf_chans  = config.get("merge_weight_clf_chans", None)
    quantize_after_simplifier  = config["quantize_after_simplifier"]
    noise_after_simplifier  = config["noise_after_simplifier"]
    soft_clamp_0_1  = config.get("soft_clamp_0_1", False)
    encoder_clip_eps  = config.get("encoder_clip_eps", 5e-2)
    if glow == "reload":
        glow_model_path_32x32 = config.get(
            "glow_model_path_32x32",
            "/home/schirrmr/data/exps/invertible-neurips/smaller-glow/21/10_model.th")
        glow = load_glow(glow_model_path_32x32)

    preproc = get_preprocessor(preproc_name, glow, encoder_clip_eps, cat_clf_chans_for_preproc,
                     merge_weight_clf_chans, unet_use_bias, quantize_after_simplifier,
                     noise_after_simplifier, soft_clamp_0_1, X)
    preproc.load_state_dict(
        th.load(os.path.join(saved_exp_folder, "preproc_state_dict.th"))
    )
    preproc.eval()
    return preproc