# fromhttps://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py
import functools
from torch import nn
import torch
import torch as th
from functools import partial


class CatExtraChannels(nn.Module):
    def __init__(self, module, n_chans, preproc_module, merge_weight_init=0.2):
        super().__init__()
        self.module = module
        self.preproc_module = preproc_module
        self.chans_to_cat = None
        self.merge_weights = nn.Parameter(th.zeros(1))
        self.merge_weights.data[:] = merge_weight_init
        self.merge_bias = nn.Parameter(th.zeros(1))

    def forward(self, x):
        assert self.chans_to_cat is not None
        chans_to_cat = self.chans_to_cat
        chans_to_cat = chans_to_cat * self.merge_weights.view(1, -1, 1, 1)
        chans_to_cat = chans_to_cat + self.merge_bias.view(1, -1, 1, 1)
        chans_to_cat = self.preproc_module(chans_to_cat)
        x = th.cat((x, chans_to_cat), dim=1)
        self.chans_to_cat = None
        return self.module(x)


class UnetGeneratorWithExtraInput(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
            self,
            input_nc,
            output_nc,
            num_downs,
            ngf=64,
            norm_layer=nn.BatchNorm2d,
            use_dropout=False,
            final_nonlin=nn.Tanh,
            nonlin_up=nn.ReLU,
            nonlin_down=partial(nn.LeakyReLU, negative_slope=0.2),
            use_bias=True,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        glow_out_shapes = [(6, 16, 16), (12, 8, 8), (48, 4, 4)]
        # construct unet structure
        # 2x2
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            use_bias=use_bias,
        )  # add the innermost layer

        # loop will beempty
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                nonlin_up=nonlin_up,
                nonlin_down=nonlin_down,
                use_bias=use_bias,
            )

        # 4x4
        # add glow channels
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=ngf * 4 + glow_out_shapes[2][0],
            submodule=unet_block,
            norm_layer=norm_layer,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )
        unet_block.model = CatExtraChannels(unet_block.model, glow_out_shapes[2][0], nn.Identity(), )
        self.block4x4 = unet_block.model

        # 8x8
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=ngf * 2 + glow_out_shapes[1][0],
            submodule=unet_block,
            norm_layer=norm_layer,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )
        unet_block.model = CatExtraChannels(unet_block.model, glow_out_shapes[1][0], nn.Identity(), )
        self.block8x8 = unet_block.model

        # 16x16
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=ngf + glow_out_shapes[0][0],
            submodule=unet_block,
            norm_layer=norm_layer,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )
        unet_block.model = CatExtraChannels(unet_block.model, glow_out_shapes[0][0])
        self.block16x16 = unet_block.model

        # 32x32
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            final_nonlin=final_nonlin,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)



class WrapResidualUnet(nn.Module):
    def __init__(self, unet, final_nonlin, add_to_init, add_during_forward):
        super().__init__()
        self.unet = unet
        self.add_during_forward = add_during_forward
        self.merge_weights = nn.Parameter(th.zeros(2, 3) + add_to_init)

        self.final_nonlin = final_nonlin

    def forward(self, x):
        unet_out = self.unet(x)
        clamped_x = th.clamp(x, 1e-2, 1 - 1e-2)
        x_inv = th.log(clamped_x) - th.log(1.0 - clamped_x)
        # if merge weights are one, it is like residual function
        merge_weights = self.merge_weights + self.add_during_forward
        merged = th.sum(
            th.stack((x_inv, unet_out), dim=1)
            * merge_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            dim=1,
        )
        out = self.final_nonlin(merged)
        return out


class WrapResidualIdentityUnet(nn.Module):
    def __init__(
        self,
        unet,
        final_nonlin=th.sigmoid,
    ):
        super().__init__()
        self.unet = unet
        self.merge_weight = nn.Parameter(th.zeros(1))

        self.final_nonlin = final_nonlin

    def forward(self, x):
        unet_out = self.unet(x)
        clamped_x = th.clamp(x, 1e-3, 1 - 1e-3)
        # invert sigmoid
        x_inv = th.log(clamped_x) - th.log(1.0 - clamped_x)
        merged = x_inv + unet_out * self.merge_weight
        out = self.final_nonlin(merged)
        return out


class WrapResidualAndBlendUnet(nn.Module):
    def __init__(
        self,
        unet,
    ):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        unet_out = self.unet(x)
        unet_out, factors = th.chunk(unet_out, 2, dim=1)
        factors = th.sigmoid(factors)
        unet_x = th.sigmoid(unet_out)
        x_inv = unet_x * factors + (1-factors) * x
        return x_inv


class WrapResidualAndMixUnet(nn.Module):
    def __init__(
        self,
        unet,
    ):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        with th.no_grad():
            clamped_x = th.clamp(x, 1e-3, 1 - 1e-3)
            # invert sigmoid
            x_inv = th.log(clamped_x) - th.log(1.0 - clamped_x)
        unet_out = self.unet(x)
        unet_out, factors_x = th.chunk(unet_out, 2, dim=1)
        out_before_sigmoid = x_inv * factors_x + unet_out# * factors_unet_x
        out = th.sigmoid(out_before_sigmoid)
        return out


class WrapResidualAndMixNonSigmoidUnet(nn.Module):
    def __init__(
        self,
        unet,
    ):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        unet_out = self.unet(x)
        unet_out, factors_x = th.chunk(unet_out, 2, dim=1)
        # restrict to desired number of channels
        out = x[:,:unet_out.shape[1]] * factors_x + unet_out  # * factors_unet_x
        return out


class WrapResidualAndMixGreyUnet(nn.Module):
    def __init__(
        self,
        unet,
    ):
        super().__init__()
        self.unet = unet

    def forward(self, x):
        with th.no_grad():
            clamped_x = th.clamp(x, 1e-3, 1 - 1e-3)
            # invert sigmoid
            x_inv = th.log(clamped_x) - th.log(1.0 - clamped_x)
        unet_out = self.unet(x)
        factors_grey_x = unet_out[:,-1:]
        unet_out, factors_x, = th.chunk(unet_out[:,:-1], 2, dim=1)
        grey_x_inv = th.mean(x_inv, dim=1, keepdim=True)
        out_before_sigmoid = x_inv * factors_x + grey_x_inv * factors_grey_x + unet_out
        out = th.sigmoid(out_before_sigmoid)
        return out


class WrapResidualNonSigmoidUnet(nn.Module):
    def __init__(
        self,
        unet,
    ):
        super().__init__()
        self.unet = unet
        self.merge_weight = nn.Parameter(th.zeros(1))

    def forward(self, x):
        unet_out = self.unet(x)
        # restrict to desired number of channels
        merged = x[:,:unet_out.shape[1]] + unet_out * self.merge_weight
        return merged

    
class UnetGeneratorCompact(nn.Module):
    """Create a Unet-based generator for less number of downs than 5"""

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        final_nonlin=nn.Tanh,
        nonlin_up=nn.ReLU,
        nonlin_down=partial(nn.LeakyReLU, negative_slope=0.2),
        use_bias=True,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            use_bias=use_bias,
        )  # add the innermost layer
        for i in range(num_downs - 2):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                nonlin_up=nonlin_up,
                nonlin_down=nonlin_down,
                use_bias=use_bias,
            )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf * 8,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            final_nonlin=final_nonlin,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
    
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        final_nonlin=nn.Tanh,
        nonlin_up=nn.ReLU,
        nonlin_down=partial(nn.LeakyReLU, negative_slope=0.2),
        use_bias=True,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            use_bias=use_bias,
        )  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                nonlin_up=nonlin_up,
                nonlin_down=nonlin_down,
                use_bias=use_bias,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4,
            ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )
        unet_block = UnetSkipConnectionBlock(
            ngf,
            ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            final_nonlin=final_nonlin,
            nonlin_up=nonlin_up,
            nonlin_down=nonlin_down,
            use_bias=use_bias,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        final_nonlin=nn.Tanh,
        nonlin_up=nn.ReLU,
        nonlin_down=partial(nn.LeakyReLU, negative_slope=0.2),
        use_bias=True,
    ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nonlin_down()
        downnorm = norm_layer(inner_nc)
        uprelu = nonlin_up()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, final_nonlin()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator8x8(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        final_nonlin=nn.Tanh,
        use_bias=True
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super().__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            use_bias=use_bias,
        )  # add the innermost layer
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
            use_bias=use_bias,
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf * 4,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            final_nonlin=final_nonlin,
            use_bias=use_bias,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
