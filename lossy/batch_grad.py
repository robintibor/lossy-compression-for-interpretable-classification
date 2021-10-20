import numpy as np
import torch
import torch as th
from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.extensions.firstorder.batch_grad import (
    batchnorm1d,
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    linear,
)
from backpack.utils.ein import eingroup
from torch.nn import (
    BatchNorm1d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Linear,
)
from torch.nn import Conv1d, Conv2d, Conv3d

from lossy.wide_nf_net import ScaledStdConv2d


class BatchGradScaledStdConv2d(FirstOrderModuleExtension):
    """Extract individual gradients for ``ScaleModule``."""

    def __init__(self):
        """Store parameters for which individual gradients should be computed."""
        # specify parameter names

        # https://github.com/f-dangel/backpack/blob/6a1ac373b01f4e4cd5d4836c0af60f30e6854abe/backpack/core/derivatives/convnd.py#L22
        self.dim_text = "x,y"
        self.conv_dims = 2
        super().__init__(
            params=[
                "gain",
                "weight",
                "bias",
            ]
        )

    def gain(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for ScaleModule's ``weight`` parameter.

        Args:
            ext(BatchGrad): extension that is used
            module(ScaleModule): module that performed forward pass
            g_inp(tuple[torch.Tensor]): input gradient tensors
            g_out(tuple[torch.Tensor]): output gradient tensors
            bpQuantities(None): additional quantities for second-order

        Returns:
            torch.Tensor: individual gradients
        """

        # hope this is correct robintibor@gmail.com
        mat = g_out[0].unsqueeze(0)
        sum_batch = False
        if module.groups != 1:
            raise NotImplementedError("Groups greater than 1 are not supported yet")

        V = mat.shape[0]
        N, C_out = module.output_shape[0], module.output_shape[1]
        C_in = module.input0_shape[1]
        C_in_axis = 1
        N_axis = 0
        dims = self.dim_text

        repeat_pattern = [1, C_in] + [1 for _ in range(self.conv_dims)]
        mat = eingroup("v,n,c,{}->vn,c,{}".format(dims, dims), mat)
        mat = mat.repeat(*repeat_pattern)
        mat = eingroup("a,b,{}->ab,{}".format(dims, dims), mat)
        mat = mat.unsqueeze(C_in_axis)

        repeat_pattern = [1, V] + [1 for _ in range(self.conv_dims)]
        input = eingroup("n,c,{}->nc,{}".format(dims, dims), module.input0)
        input = input.unsqueeze(N_axis)
        input = input.repeat(*repeat_pattern)

        grad_weight = th.nn.functional.conv2d(
            input,
            mat,
            bias=None,
            stride=module.dilation,
            padding=module.padding,
            dilation=module.stride,
            groups=C_in * N * V,
        ).squeeze(0)

        for dim in range(self.conv_dims):
            axis = dim + 1
            size = module.weight.shape[2 + dim]
            grad_weight = grad_weight.narrow(axis, 0, size)

        sum_dim = "" if sum_batch else "n,"
        eingroup_eq = "vnio,{}->v,{}o,i,{}".format(dims, sum_dim, dims)

        grad_on_computed_weight = eingroup(
            eingroup_eq, grad_weight, dim={"v": V, "n": N, "i": C_in, "o": C_out}
        )
        module.grad_on_computed_weight = grad_on_computed_weight
        computed_weight_before_gain = module.get_weight() / module.gain
        grad_gain = grad_on_computed_weight.squeeze(0) * computed_weight_before_gain
        grad_gain = th.sum(grad_gain, dim=(2, 3, 4), keepdim=True)
        return grad_gain

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for ScaleModule's ``weight`` parameter.

        Args:
            ext(BatchGrad): extension that is used
            module(ScaleModule): module that performed forward pass
            g_inp(tuple[torch.Tensor]): input gradient tensors
            g_out(tuple[torch.Tensor]): output gradient tensors
            bpQuantities(None): additional quantities for second-order

        Returns:
            torch.Tensor: individual gradients
        """
        grad_on_computed_weight = module.grad_on_computed_weight

        del module.grad_on_computed_weight

        grad_weight_before_gain = grad_on_computed_weight.squeeze(0) * module.gain
        grad_weight_before_scale = grad_weight_before_gain * module.scale
        grad_weight_before_mean = grad_weight_before_scale - th.mean(
            grad_weight_before_scale, dim=(2, 3, 4), keepdim=True
        )

        # Account for std transformation
        grad_outer_w = grad_weight_before_mean
        eps = module.eps
        w = module.weight
        n_params_per_w = np.prod(w.shape[1:])
        flat_w = th.flatten(w, start_dim=1)
        n_x = len(grad_outer_w)
        grad_outer_w_flat = th.flatten(grad_outer_w, start_dim=2)
        std = torch.std(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        grad_w_s = (
            0.5
            * (2 * w - 2 * th.mean(w, dim=(1, 2, 3), keepdim=True))
            / (n_params_per_w * th.sqrt(std * std))
        )
        grad_w_1_div_s = -(1 / ((std + eps) * (std + eps))) * grad_w_s
        grad_w_1_div_s_flat = th.flatten(grad_w_1_div_s, start_dim=1)

        # unsqueeze(0) for example dim
        grad_on_factor_per_w_p = grad_outer_w_flat * flat_w.unsqueeze(0)

        grad_on_w_through_std = (
            grad_on_factor_per_w_p.sum(dim=2, keepdim=True) * grad_w_1_div_s_flat
        ).reshape(n_x, *w.shape)

        grad_on_w_direct = grad_outer_w * (1 / (std.unsqueeze(0) + eps))
        grad_weight = grad_on_w_direct + grad_on_w_through_std

        return grad_weight

    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        """Extract individual gradients for ScaleModule's ``weight`` parameter.

        Args:
            ext(BatchGrad): extension that is used
            module(ScaleModule): module that performed forward pass
            g_inp(tuple[torch.Tensor]): input gradient tensors
            g_out(tuple[torch.Tensor]): output gradient tensors
            bpQuantities(None): additional quantities for second-order

        Returns:
            torch.Tensor: individual gradients
        """
        mat = g_out[0]
        # had to be changed from original, not sure what the additional axis was for
        # original: https://github.com/f-dangel/backpack/blob/6a1ac373b01f4e4cd5d4836c0af60f30e6854abe/backpack/core/derivatives/convnd.py#L96-L100
        # axes = list(range(3, len(module.output_shape) + 1))
        axes = list(range(2, len(module.output_shape)))
        return mat.sum(axes)


class BatchGradNFNets(BackpropExtension):
    """Individual gradients for each sample in a minibatch.
    Stores the output in ``grad_batch`` as a ``[N x ...]`` tensor,
    where ``N`` batch size and ``...`` is the shape of the gradient.
    Note: beware of scaling issue
        The `individual gradients` depend on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``BatchGrad`` will return
        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.
    The concept of individual gradients is only meaningful if the
    objective is a sum of independent functions (no batchnorm).
    """

    def __init__(self):
        super().__init__(
            savefield="grad_batch",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.BatchGradLinear(),
                Conv1d: conv1d.BatchGradConv1d(),
                Conv2d: conv2d.BatchGradConv2d(),
                Conv3d: conv3d.BatchGradConv3d(),
                ScaledStdConv2d: BatchGradScaledStdConv2d(),
                ConvTranspose1d: conv_transpose1d.BatchGradConvTranspose1d(),
                ConvTranspose2d: conv_transpose2d.BatchGradConvTranspose2d(),
                ConvTranspose3d: conv_transpose3d.BatchGradConvTranspose3d(),
                BatchNorm1d: batchnorm1d.BatchGradBatchNorm1d(),
            },
        )
