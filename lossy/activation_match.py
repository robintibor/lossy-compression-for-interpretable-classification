from copy import copy
import torch as th
import numpy as np
from einops import rearrange
from functools import partial
from torch import nn
from typing import List
from torch import Tensor
import torch.nn.functional as F


def restore_grads_from(params, fieldname):
    for p in params:
        p.grad = getattr(p, fieldname)  # .detach().clone()
        delattr(p, fieldname)


def save_grads_to(params, fieldname):
    for p in params:
        setattr(p, fieldname, p.grad.detach().clone())

# https://github.com/f-dangel/backpack/blob/1da7e53ebb2c490e2b7dd9f79116583641f3cca1/backpack/utils/subsampling.py
def subsample(tensor: Tensor, dim: int = 0, subsampling: List[int] = None) -> Tensor:
    """Select samples from a tensor along a dimension.
    Args:
        tensor: Tensor to select from.
        dim: Selection dimension. Defaults to ``0``.
        subsampling: Indices of samples that are sliced along the dimension.
            Defaults to ``None`` (use all samples).
    Returns:
        Tensor of same rank that is sub-sampled along the dimension.
    """
    if subsampling is None:
        return tensor
    else:
        return tensor[(slice(None),) * dim + (subsampling,)]


class ReLUSoftPlusGrad(nn.Module):
    def __init__(self, softplus_mod):
        super().__init__()
        self.softplus = softplus_mod

    def forward(self, x):
        relu_x = nn.functional.relu(x)
        softplus_x = self.softplus(x)
        return softplus_x + (relu_x - softplus_x).detach()


def get_all_activations(
    net,
    X,
    wanted_modules=None,
):
    if wanted_modules is None:
        wanted_modules = net.modules()
    activations = []

    def append_activations(module, input, output):
        activations.append(output)

    handles = []
    for module in wanted_modules:
        handle = module.register_forward_hook(append_activations)
        handles.append(handle)
    try:
        _ = net(X)
    finally:
        for h in handles:
            h.remove()
    return activations


def get_in_out_acts_and_in_out_grads_per_module(
    net, X, loss_fn, wanted_modules=None, data_parallel=False,
        **backward_kwargs
):
    if wanted_modules is None:
        wanted_modules = net.modules()

    if data_parallel:
        module_to_vals = {}
    else:
        module_to_vals = {m: {} for m in wanted_modules}

    handles = []

    def append_grads(module, grad_input, grad_output):
        if grad_output is not None:
            if "out_grad" not in module_to_vals[module]:
                module_to_vals[module]["out_grad"] = []
            module_to_vals[module]["out_grad"].append(grad_output)
        else:
            assert grad_input is not None
            if "in_grad" not in module_to_vals[module]:
                module_to_vals[module]["in_grad"] = []
            module_to_vals[module]["in_grad"].append(grad_input)

    def append_activations(module, input, output):
        if data_parallel:
            module_to_vals[module] = {}
        else:
            assert "in_act" not in module_to_vals[module]
            assert "out_act" not in module_to_vals[module]
        module_to_vals[module]["in_act"] = input
        if hasattr(output, "register_hook"):
            output = (output,)
        module_to_vals[module]["out_act"] = output

        for a_output in output:
            if a_output is None:
                continue
            # see https://github.com/pytorch/pytorch/issues/25723 for why like this
            handle = a_output.register_hook(
                partial(
                    append_grads,
                    module,
                    None,
                )
            )
            handles.append(handle)
        for a_input in input:
            # see https://github.com/pytorch/pytorch/issues/25723 for why like this
            handle = a_input.register_hook(
                partial(
                    append_grads,
                    module,
                    grad_output=None,
                )
            )
            handles.append(handle)

    for module in wanted_modules:
        handle = module.register_forward_hook(append_activations)
        handles.append(handle)
    try:
        output = net(X)
        loss = loss_fn(output)
        loss.backward(**backward_kwargs)
    finally:
        for h in handles:
            h.remove()
    return module_to_vals


def get_in_out_activations_per_module(net, X, wanted_modules=None, data_parallel=False):
    if wanted_modules is None:
        wanted_modules = net.modules()

    if data_parallel:
        module_to_vals = {}
    else:
        module_to_vals = {m: {} for m in wanted_modules}

    def append_activations(module, input, output):
        if data_parallel:
            module_to_vals[module] = {}
        else:
            assert "in_act" not in module_to_vals[module]
            assert "out_act" not in module_to_vals[module]
        module_to_vals[module]["in_act"] = input
        if hasattr(output, "register_hook"):
            output = (output,)
        module_to_vals[module]["out_act"] = output

    handles = []
    for module in wanted_modules:
        handle = module.register_forward_hook(append_activations)
        handles.append(handle)
    try:
        _ = net(X)
    finally:
        for h in handles:
            h.remove()
    return module_to_vals


def clip_min_max_signed(a, b):
    mask = b >= 0
    c = mask * th.minimum(a, b) + (~mask) * th.maximum(a, b)
    return c


def conv_grad_groups(module, in_act, out_grad):
    weight_bgrad, bias_bgrad = conv_backward(
        in_act,
        out_grad,
        module.in_channels,
        module.out_channels,
        module.kernel_size,
        bias=module.bias is not None,
        stride=module.stride,
        dilation=module.dilation,
        padding=module.padding,
        groups=module.groups,
        nd=2,
    )
    return weight_bgrad, bias_bgrad


def scaled_conv_grad(module, in_act, out_grad, conv_grad_fn):
    grad_on_computed_weight, grad_on_bias = conv_batch_grad(
        module, in_act, out_grad, conv_grad_fn=conv_grad_fn)
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
    std = th.std(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
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

    computed_weight_before_gain = module.get_weight() / module.gain
    grad_gain = grad_on_computed_weight.squeeze(0) * computed_weight_before_gain
    grad_gain = th.sum(grad_gain, dim=(2, 3, 4), keepdim=True)
    return grad_weight, grad_on_bias, grad_gain


# https://github.com/owkin/grad-cnns/blob/b0a9e3bb16f6a2358d3d8e9c936d8d308a648476/code/gradcnn/crb_backward.py#L25
def conv_backward(
    input,
    grad_output,
    in_channels,
    out_channels,
    kernel_size,
    bias=True,
    stride=1,
    dilation=1,
    padding=0,
    groups=1,
    nd=1,
):
    """Computes per-example gradients for nn.Conv1d and nn.Conv2d layers.
    This function is used in the internal behaviour of bnn.Linear.
    """

    # Change format of stride from int to tuple if necessary.
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    if isinstance(stride, int):
        stride = (stride,) * nd
    if isinstance(dilation, int):
        dilation = (dilation,) * nd
    if isinstance(padding, int):
        padding = (padding,) * nd
    if isinstance(padding, str):
        if padding == 'same':
            padding = tuple(np.array(kernel_size) // 2)
        elif padding == 'valid':
            padding (0,) * nd
        else:
            raise ValueError(f"Unknown padding {padding:s}")

    # Get some useful sizes
    batch_size = input.size(0)
    input_shape = input.size()[-nd:]
    output_shape = grad_output.size()[-nd:]

    # Reshape to extract groups from the convolutional layer
    # Channels are seen as an extra spatial dimension with kernel size 1
    input_conv = input.view(1, batch_size * groups, in_channels // groups, *input_shape)

    # Compute convolution between input and output; the batchsize is seen
    # as channels, taking advantage of the `groups` argument
    grad_output_conv = grad_output.view(-1, 1, 1, *output_shape)

    stride = (1, *stride)
    dilation = (1, *dilation)
    padding = (0, *padding)

    if nd == 1:
        convnd = F.conv2d
        s_ = np.s_[..., : kernel_size[0]]
    elif nd == 2:
        convnd = F.conv3d
        s_ = np.s_[..., : kernel_size[0], : kernel_size[1]]
    elif nd == 3:
        raise NotImplementedError(
            "3d convolution is not available with current per-example gradient computation"
        )

    conv = convnd(
        input_conv,
        grad_output_conv,
        groups=batch_size * groups,
        stride=dilation,
        dilation=stride,
        padding=padding,
    )

    # Because of rounding shapes when using non-default stride or dilation,
    # convolution result must be truncated to convolution kernel size
    conv = conv[s_]

    # Reshape weight gradient to correct shape
    new_shape = [batch_size, out_channels, in_channels // groups, *kernel_size]
    weight_bgrad = conv.view(*new_shape).contiguous()

    # Compute bias gradient
    grad_output_flat = grad_output.view(batch_size, grad_output.size(1), -1)
    bias_bgrad = th.sum(grad_output_flat, dim=2) if bias else None

    return weight_bgrad, bias_bgrad


def conv_batch_grad(module, in_act, out_grad, conv_grad_fn='loop'):
    #assert np.all(np.array(module.kernel_size) // 2 == np.array(module.padding))
    # recreated_weight_batch_grad = th.nn.functional.conv2d(
    #     in_act.view(-1, 1, *in_act.shape[2:]),
    #     out_grad.view(-1, 1, *out_grad.shape[2:]),
    #     padding=module.padding,
    #     dilation=module.stride,
    # )
    # reshaped_weight_batch_grad = recreated_weight_batch_grad.reshape(
    #     len(in_act), in_act.shape[1], *recreated_weight_batch_grad.shape[1:]
    # ).transpose(1, 2)
    # # may be stride meant last kernel filter did not fit inside input
    # # then have to remove appropriately
    # reshaped_weight_batch_grad = reshaped_weight_batch_grad[
    #     :, :, :, : module.weight.shape[2], : module.weight.shape[3]
    # ]
    # # n_out_before_stride = np.array(in_act.shape[2:]) - (
    # #        np.array(module.kernel_size) + 1 + np.array(module.padding) * 2)
    # # n_removed_by_stride =
    if conv_grad_fn == 'loop':
        weight_batch_grad = conv_weight_grad_loop(
            module, in_act, out_grad)
    elif conv_grad_fn == 'backpack':
        weight_batch_grad = conv_weight_grad_backpack(module, in_act, out_grad)

    bias_batch_grad = out_grad.sum(dim=(-2, -1))
    #print("hi")
    #weight_batch_grad, bias_batch_grad = conv_grad_groups(module, in_act, out_grad)
    return weight_batch_grad, bias_batch_grad


def add_conv_bias_grad(conv_weight_grad_fn):
    def conv_grad_fn(module, in_act, out_grad):
        weight_grad = conv_weight_grad_fn(module, in_act, out_grad)
        bias_grad = out_grad.sum(dim=(-2, -1))
        return weight_grad, bias_grad

    return conv_grad_fn


def conv_weight_grad_backpack(module, in_act, out_grad):
    mat = out_grad.unsqueeze(0)
    G = module.groups
    V = mat.shape[0]
    C_out = out_grad.shape[1]
    N = out_grad.shape[0]
    C_in = in_act.shape[1]
    C_in_axis = 1
    N_axis = 0
    sum_batch = False
    dims = "x y"
    conv_dims = 2

    # treat channel groups like vectorization (v) and batch (n) axes
    mat = rearrange(mat, "v n (g c) ... -> (v n g) c ...", g=G, c=C_out // G)

    repeat_pattern = [1, C_in // G] + [1 for _ in range(conv_dims)]
    mat = mat.repeat(*repeat_pattern)
    mat = rearrange(mat, "a b ... -> (a b) ...")
    mat = mat.unsqueeze(C_in_axis)

    input = rearrange(subsample(in_act, subsampling=None), "n c ... -> (n c) ...")
    input = input.unsqueeze(N_axis)
    repeat_pattern = [1, V] + [1 for _ in range(conv_dims)]
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

    for dim in range(conv_dims):
        axis = dim + 1
        size = module.weight.shape[2 + dim]
        grad_weight = grad_weight.narrow(axis, 0, size)

    dim = {"g": G, "v": V, "n": N, "i": C_in // G, "o": C_out // G}
    if sum_batch:
        grad_weight = reduce(
            grad_weight, "(v n g i o) ... -> v (g o) i ...", "sum", **dim
        )
    else:
        grad_weight = rearrange(
            grad_weight, "(v n g i o) ... -> v n (g o) i ...", **dim
        )
    return grad_weight.squeeze(0)


def conv_weight_grad_loop(module, in_act, out_grad):
    grads = [conv_weight_grad(
            module, in_act[i_ex : i_ex + 1], out_grad[i_ex : i_ex + 1]
        ) for i_ex in range(len(in_act))]
    return th.stack(grads)


def conv_weight_grad(module, in_act, out_grad):
    padding = module.padding
    if isinstance(padding, int):
        padding = (padding,) * (in_act.ndim - 2)
    if isinstance(padding, str):
        if padding == 'same':
            padding = tuple(np.array(module.kernel_size) // 2)
        elif padding == 'valid':
            padding = (0,) * (in_act.ndim - 2)
        else:
            raise ValueError(f"Unknown padding {padding:s}")
    recreated_weight_grad = th.nn.functional.conv2d(
        in_act.transpose(0, 1),
        out_grad.transpose(0, 1),
        padding=padding,
        dilation=module.stride,
    )
    reshaped_weight_grad = recreated_weight_grad.transpose(0, 1)
    # may be stride meant last kernel filter did not fit inside input
    # then have to remove appropriately
    reshaped_weight_grad = reshaped_weight_grad[
        :, :, : module.weight.shape[2], : module.weight.shape[3]
    ]
    return reshaped_weight_grad


def linear_batch_grad(module, in_act, out_grad):
    weight_batch_grad = in_act.unsqueeze(1) * out_grad.unsqueeze(2)
    bias_batch_grad = out_grad
    return weight_batch_grad, bias_batch_grad


def bnorm_batch_grad(module, in_act, out_grad):
    bn = module
    assert not bn.training
    unnormed_in = in_act
    n_examples = unnormed_in.shape[0]
    n_chans = unnormed_in.shape[1]
    demeaned_in = unnormed_in - bn.running_mean.view(1, n_chans, 1, 1)
    # th.mean(unnormed_in, dim=(0,-2,-1), keepdim=True)
    normed_in = demeaned_in / (bn.running_var.sqrt().view(1, n_chans, 1, 1) + bn.eps)
    # (th.std(demeaned_in, dim=(0,-2,-1), keepdim=True) +  bn.eps)
    weight_batch_grad = th.sum(normed_in * out_grad, dim=(-2, -1))
    bias_batch_grad = out_grad.sum(dim=(-2, -1))
    return weight_batch_grad, bias_batch_grad


def relued(val_func):
    def relued_val_func(m, x_m_vals, ref_m_vals):
        x_and_ref_vals = val_func(m, x_m_vals, ref_m_vals)
        relued_x_and_ref_vals = []
        for x_val, ref_val in x_and_ref_vals:
            mask = ref_val > 0
            x_val = x_val * mask
            ref_val = ref_val * mask
            relued_x_and_ref_vals.append((x_val, ref_val))
        return relued_x_and_ref_vals

    return relued_val_func


def clipped(val_func):
    def clipped_val_func(m, x_m_vals, ref_m_vals):
        x_and_ref_vals = val_func(m, x_m_vals, ref_m_vals)
        clipped_x_and_ref_vals = []
        for x_val, ref_val in x_and_ref_vals:
            x_val = th.minimum(x_val, ref_val)
            clipped_x_and_ref_vals.append((x_val, ref_val))
        return clipped_x_and_ref_vals

    return clipped_val_func


def refed(val_func, key):
    def refed_val_func(m, x_m_vals, ref_m_vals):
        x_m_vals = copy(x_m_vals)
        x_m_vals[key] = ref_m_vals[key]
        return val_func(m, x_m_vals, ref_m_vals)

    return refed_val_func


def grad_out_act(m, x_m_vals, ref_m_vals):
    return list(zip(x_m_vals["out_grad"], ref_m_vals["out_grad"]))


def grad_in_act(m, x_m_vals, ref_m_vals):
    return list(zip(x_m_vals["in_grad"], ref_m_vals["in_grad"]))

def in_act(m, x_m_vals, ref_m_vals):
    return list(zip(x_m_vals["in_act"], ref_m_vals["in_act"]))

def out_act(m, x_m_vals, ref_m_vals):
    return list(zip(x_m_vals["out_act"], ref_m_vals["out_act"]))


def grad_out_act_act(m, x_m_vals, ref_m_vals):
    return [
        (x_a * x_g, r_a * r_g)
        for x_a, x_g, r_a, r_g in zip(
            x_m_vals["out_act"],
            x_m_vals["out_grad"],
            ref_m_vals["out_act"],
            ref_m_vals["out_grad"],
        )
    ]


def grad_act_relued(m, x_m_vals, ref_m_vals):
    vals = []
    for x_a, x_g, r_a, r_g in zip(
        x_m_vals["out_act"],
        x_m_vals["out_grad"],
        ref_m_vals["out_act"],
        ref_m_vals["out_grad"],
    ):
        mask = (r_a * r_g) > 0
        vals.append((mask * x_g), (mask * r_g))
    return vals


def grad_in_act_act(m, x_m_vals, ref_m_vals):
    return [
        (x_a * x_g, r_a * r_g)
        for x_a, x_g, r_a, r_g in zip(
            x_m_vals["in_act"],
            x_m_vals["in_grad"],
            ref_m_vals["in_act"],
            ref_m_vals["in_grad"],
        )
    ]


def grad_in_act_act_same_sign(m, x_m_vals, ref_m_vals):
    vals = []
    for x_a, x_g, r_a, r_g in zip(
        x_m_vals["in_act"],
        x_m_vals["in_grad"],
        ref_m_vals["in_act"],
        ref_m_vals["in_grad"],
    ):
        mask_ref = (r_a * r_g) > 0
        mask = mask_ref * (x_g.sign() == r_g.sign())

        x_val = r_g * x_a  # clip_min_max(xg, rg) *
        ref_val = mask * r_g * r_a
        vals.append((x_val, ref_val))
    return vals


def grad_in_act_act_same_sign_masked(m, x_m_vals, ref_m_vals):
    vals = []
    for x_a, x_g, r_a, r_g in zip(
        x_m_vals["in_act"],
        x_m_vals["in_grad"],
        ref_m_vals["in_act"],
        ref_m_vals["in_grad"],
    ):
        mask_ref = (r_a * r_g) > 0
        mask = mask_ref * (x_g.sign() == r_g.sign())

        x_val = r_g * x_a * mask  # clip_min_max(xg, rg) *
        ref_val = mask * r_g * r_a
        vals.append((x_val, ref_val))
    return vals


def unfolded_grads(m, x_m_vals, ref_m_vals, conv_grad_fn=conv_grad_groups, renorm_grads=False):
    assert m.__class__.__name__ in [
        "BatchNorm2d",
        "Conv2d",
        "Linear",
        "ScaledStdConv2d",
    ], f"Unsupported class {m.__class__.__name__}"
    grad_fn = {
        "ScaledStdConv2d": conv_grad_fn,  # conv_grad_groups,  # for now just match grad before weight scaling
        "BatchNorm2d": bnorm_batch_grad,
        "Conv2d": conv_grad_fn,
        "Linear": linear_batch_grad,
    }[m.__class__.__name__]

    eps = 1e-20
    ref_square_grads = th.square(ref_m_vals["out_grad"][0])
    sum_squared_out_grads = th.sum(
        ref_square_grads, dim=tuple(range(1, ref_square_grads.ndim)), keepdim=True) + eps
    ref_out_grads = ref_m_vals["out_grad"][0]
    simple_out_grads = x_m_vals["out_grad"][0]
    if renorm_grads:
        ref_out_grads = ref_out_grads / th.sqrt(sum_squared_out_grads)
        simple_out_grads = simple_out_grads / th.sqrt(sum_squared_out_grads)
        ref_square_grads = ref_square_grads / sum_squared_out_grads


    w_g_x_2, b_g_x_2 = grad_fn(
        m, th.square(x_m_vals["in_act"][0]), th.square(simple_out_grads)
    )
    w_g_r_2, b_g_r_2 = grad_fn(
        m, th.square(ref_m_vals["in_act"][0]), ref_square_grads
    )
    w_g_x_r, b_g_x_r = grad_fn(
        m,
        x_m_vals["in_act"][0] * ref_m_vals["in_act"][0],
        simple_out_grads * ref_out_grads,
    )

    return (
        (
            (
                w_g_x_2,
                w_g_x_r,
            ),
            (w_g_r_2, w_g_x_r),
        ),
        (
            (
                b_g_x_2,
                b_g_x_r,
            ),
            (b_g_r_2, b_g_x_r),
        ),
    )


def cos_dist_unfolded_grads(w_x_tuple, w_r_tuple, eps):
    (w_g_x_2, w_g_x_r) = w_x_tuple
    (w_g_r_2, w_g_x_r) = w_r_tuple
    cos_dist = 1 - (
        th.sum(th.flatten(w_g_x_r, 1), 1)
        / (
            th.sqrt(th.sum(th.flatten(w_g_x_2, 1), 1))
            * th.sqrt(th.sum(th.flatten(w_g_r_2, 1), 1))
            + eps
        )
    )
    return cos_dist


def gradparam_per_batch(m, vals, conv_grad_fn):
    assert m.__class__.__name__ in ["BatchNorm2d", "Conv2d", "Linear", "ScaledStdConv2d"]
    grad_fn = {
        "BatchNorm2d": bnorm_batch_grad,
        "Conv2d": partial(conv_batch_grad, conv_grad_fn=conv_grad_fn),
        "Linear": linear_batch_grad,
        "ScaledStdConv2d": partial(scaled_conv_grad, conv_grad_fn=conv_grad_fn),
    }[m.__class__.__name__]

    grad_tuple = grad_fn(
        m, vals["in_act"][0], vals["out_grad"][0]
    )
    assert len(grad_tuple) in [2,3]
    grads = {}
    if m.weight is not None:
        grads["weight"] = grad_tuple[0]
    if m.bias is not None:
        grads["bias"] = grad_tuple[1]
    if hasattr(m, 'gain') and m.gain is not None:
        grads["gain"] = grad_tuple[2]
    return grads


def gradparam(m, x_m_vals, ref_m_vals):
    assert m.__class__.__name__ in ["BatchNorm2d", "Conv2d", "Linear"]
    x_grads = gradparam_per_batch(m, x_m_vals)
    ref_grads = gradparam_per_batch(m, ref_m_vals)
    grad_tuples = [(x_grads[key], ref_grads[key]) for key in x_grads]
    return grad_tuples


def gradparam_relued(m, x_m_vals, ref_m_vals):
    assert m.__class__.__name__ in ["ScaledStdConv2d", "BatchNorm2d", "Conv2d", "Linear"]
    x_grads = gradparam_per_batch(m, x_m_vals)
    ref_grads = gradparam_per_batch(m, ref_m_vals)

    grad_tuples = []
    for key in x_grads:
        param = getattr(m, key).unsqueeze(0)
        mask = (ref_grads[key] * param) > 0
        x_val = mask * x_grads[key]
        ref_val = mask * ref_grads[key]
        grad_tuples.append((x_val, ref_val))

    return grad_tuples


def gradparam_param(m, x_m_vals, ref_m_vals, conv_grad_fn):
    assert m.__class__.__name__ in ["ScaledStdConv2d", "BatchNorm2d", "Conv2d", "Linear"]
    x_grads = gradparam_per_batch(m, x_m_vals, conv_grad_fn=conv_grad_fn)
    ref_grads = gradparam_per_batch(m, ref_m_vals, conv_grad_fn=conv_grad_fn)

    grad_tuples = [
        (
            x_grads[key] * getattr(m, key).unsqueeze(0),
            ref_grads[key] * getattr(m, key).unsqueeze(0),
        )
        for key in x_grads
    ]
    return grad_tuples


def relu_match(val_func):
    def relued_val_func(m, x_m_vals, ref_m_vals):
        x_and_ref_vals = val_func(m, x_m_vals, ref_m_vals)
        x_and_ref_vals = [
            (x_val, th.nn.functional.relu(ref_val)) for x_val, ref_val in x_and_ref_vals
        ]
        return x_and_ref_vals

    return relued_val_func


def clip_nonzero_max(val_func):
    def clipped_val_func(m, x_m_vals, ref_m_vals):
        x_and_ref_vals = val_func(m, x_m_vals, ref_m_vals)
        x_and_ref_vals = [
            (th.where(ref_val > 0, th.minimum(x_val, ref_val), x_val), ref_val)
            for x_val, ref_val in x_and_ref_vals
        ]
        return x_and_ref_vals

    return clipped_val_func


def gradparam_param_unfolded(m, x_m_vals, ref_m_vals):
    if m.__class__.__name__ == "Conv2d":
        x_grad_ws = unfolded_w_grad_w(m, x_m_vals["in_act"][0], x_m_vals["out_grad"][0])
        with th.no_grad():
            ref_grad_ws = unfolded_w_grad_w(
                m, ref_m_vals["in_act"][0], ref_m_vals["out_grad"][0]
            )
        return x_grad_ws, ref_grad_ws
    else:
        return gradparam_param(m, x_m_vals, ref_m_vals)


def unfolded_w_grad_w(
    m,
    in_act,
    out_grad,
):
    all_grad_ws = []
    padded_in = th.nn.functional.pad(in_act, m.padding * 2)
    for i_h in range(m.weight.shape[2]):
        for i_w in range(m.weight.shape[3]):
            part_in = padded_in[
                :,
                :,
                i_h : i_h + m.stride[0] * out_grad.shape[2] : m.stride[0],
                i_w : i_w + m.stride[1] * out_grad.shape[2] : m.stride[1],
            ]
            out = part_in.unsqueeze(1) * out_grad.unsqueeze(2)
            out = out * m.weight.unsqueeze(0)[:, :, :, i_h : i_h + 1, i_w : i_w + 1]
            all_grad_ws.append(out)
    all_grad_ws = th.stack(all_grad_ws, dim=1)
    return all_grad_ws


def compute_dist(
    dist_fn,
    val_fn,
    X_act_grads,
    ref_act_grads,
    flatten_before=True,
    per_module=False,
    per_model=False,

):
    per_val = (not per_module) and (not per_model)
    dists = []
    val_x_all = []
    val_ref_all = []
    for m in X_act_grads:
        vals = val_fn(m, X_act_grads[m], ref_act_grads[m])
        val_x_for_module = []
        val_ref_for_module = []
        for val_x, val_ref in vals:
            if flatten_before:
                val_x = th.flatten(val_x, start_dim=1)
                val_ref = th.flatten(val_ref, start_dim=1)
            if per_val:
                dist = dist_fn(val_x, val_ref)
                dists.append(dist)
            if per_module:
                val_x_for_module.append(val_x)
                val_ref_for_module.append(val_ref)
            if per_model:
                val_x_all.append(val_x)
                val_ref_all.append(val_ref)
        if per_module:
            dist = dist_fn(th.cat(val_x_for_module, dim=1),
                           th.cat(val_ref_for_module, dim=1))
            dists.append(dist)
    if per_model:
        dist = dist_fn(th.cat(val_x_all, dim=1),
                       th.cat(val_ref_all, dim=1))
        dists.append(dist)

    return dists


def compute_vals(
    val_fn,
    X_act_grads,
    ref_act_grads,
):
    all_vals = [val_fn(m, X_act_grads[m], ref_act_grads[m]) for m in X_act_grads]
    return all_vals


def compute_multiple_dists(dist_fns, val_fn, X_act_grads, ref_act_grads):
    dists_per_fn = {d_fn: [] for d_fn in dist_fns}
    for m in X_act_grads:
        vals = val_fn(m, X_act_grads[m], ref_act_grads[m])
        for val_x, val_ref in vals:
            for dist_fn in dist_fns:
                dist = dist_fn(
                    th.flatten(val_x, start_dim=1), th.flatten(val_ref, start_dim=1)
                )
                dists_per_fn[dist_fn].append(dist)
    return dists_per_fn


def cosine_distance(*args, **kwargs):
    return 1 - th.nn.functional.cosine_similarity(*args, **kwargs)


def sse_loss(x, *args, **kwargs):
    return th.nn.functional.mse_loss(x, *args, **kwargs, reduction="none").sum(
        dim=tuple(range(1, len(x.shape)))
    )


def mse_loss(x, *args, **kwargs):
    return th.nn.functional.mse_loss(x, *args, **kwargs, reduction="none").mean(
        dim=tuple(range(1, len(x.shape)))
    )


def l1_dist(a,b,):
    return th.sum(th.abs(a - b), dim=tuple(range(1, len(a.shape))))


def asym_cos_dist(val_x, val_ref):
    return 1 - (th.sum(val_x * val_ref, dim=1) / th.sum(val_ref * val_ref, dim=1))


def larger_magnitude_cos_dist(val_x, val_ref, *args, **kwargs):
    cos_sim = th.nn.functional.cosine_similarity(val_x, val_ref)
    ref_sqrt_squared_sum = th.sqrt(th.sum(val_ref * val_ref, dim=1))
    x_sqrt_squared_sum = th.sqrt(th.sum(val_x * val_x, dim=1))
    ratios = x_sqrt_squared_sum / ref_sqrt_squared_sum
    assert ratios.shape == cos_sim.shape
    corrected_cos_sim = th.clamp_max(ratios, 1) * cos_sim
    return 1 - corrected_cos_sim


def normed_mse(val_x, val_ref, *args, **kwargs):
    mses = th.nn.functional.mse_loss(
        val_x, val_ref, *args, **kwargs, reduction="none"
    ).mean(dim=tuple(range(1, len(val_x.shape))))
    return mses / th.mean(val_ref * val_ref, dim=tuple(range(1, len(val_x.shape))))


def normed_sse(x_vals, ref_vals, eps=1e-15):
    diffs = th.sum(th.square(x_vals - ref_vals),
                   dim=tuple(range(1, len(x_vals.shape))))
    sum_squares = th.sum(th.square(ref_vals),
                         dim=tuple(range(1, len(x_vals.shape))))
    return diffs / (sum_squares + eps)


def normed_l1(x_vals, ref_vals, eps=1e-15):
    diffs = th.sum(th.abs(x_vals - ref_vals),
                   dim=tuple(range(1, len(x_vals.shape))))
    sum_abs = th.sum(th.abs(ref_vals),
                         dim=tuple(range(1, len(x_vals.shape))))
    return diffs / (sum_abs + eps)


def normed_sqrt_sse(x_vals, ref_vals, eps=1e-15):
    diffs = th.sqrt(th.sum(th.square(x_vals - ref_vals),
                   dim=tuple(range(1, len(x_vals.shape)))))
    sum_squares = th.sqrt(th.sum(th.square(ref_vals),
                         dim=tuple(range(1, len(x_vals.shape)))) + eps)
    return diffs / sum_squares


def detach_acts_grads(acts_grads):
    for m in acts_grads:
        for key in acts_grads[m]:
            acts_grads[m][key] = [
                a.detach() if hasattr(a, "detach") else a for a in acts_grads[m][key]
            ]
    return acts_grads


def filter_act_grads(act_grads, wanted_keys):
    filtered = {}
    for m in act_grads:
        if all([key in act_grads[m] for key in wanted_keys]):
            filtered[m] = act_grads[m]
    return filtered


def update_square_and_mean_grads(
    grads_per_module, step, mean_dict, square_dict, beta1, beta2
):
    with th.no_grad():
        for m in grads_per_module:
            grads_for_module = grads_per_module[m]
            for k in grads_for_module:
                new_grad = grads_for_module[k].sum(dim=0)
                old_mean = mean_dict[getattr(m, k)]
                assert new_grad.shape == old_mean.shape
                old_mean.mul_(beta1).add_(
                    new_grad,
                    alpha=1 - beta1,
                )
                # old_mean.div_(1 - (beta1 ** step))

                # compute estimate of squares
                new_mean_of_squares = grads_for_module[k].square().sum(dim=0)
                old_mean_of_squares = square_dict[getattr(m, k)]
                assert new_mean_of_squares.shape == old_mean_of_squares.shape
                old_mean_of_squares.mul_(beta2).add_(
                    new_mean_of_squares,
                    alpha=1 - beta2,
                )
                # old_mean_of_squares.div_(1 - (beta2 ** step))


def compute_square_and_mean_grads_with_grad(
    grads_per_module, step, mean_dict, square_dict, beta1, beta2
):
    temp_opt_state = {"simple_square": {}, "simple_mean": {}}
    for m in grads_per_module:
        grads_for_module = grads_per_module[m]
        for k in grads_for_module:
            new_grad = grads_for_module[k].sum(dim=0)
            old_mean = mean_dict[getattr(m, k)].detach()
            assert new_grad.shape == old_mean.shape
            new_mean = beta1 * old_mean + (1 - beta1) * new_grad
            # new_mean = new_mean / (1 - (beta1 ** step))
            temp_opt_state["simple_mean"][getattr(m, k)] = new_mean

            # compute estimate of squares
            new_mean_of_squares = grads_for_module[k].square().sum(dim=0)
            old_mean_of_squares = square_dict[getattr(m, k)].detach()
            assert new_mean_of_squares.shape == old_mean_of_squares.shape
            new_mean_of_squares = (
                beta2 * old_mean_of_squares + (1 - beta2) * new_mean_of_squares
            )
            # new_mean_of_squares = new_mean_of_squares / (1 - (beta2 ** step))
            temp_opt_state["simple_square"][getattr(m, k)] = new_mean_of_squares

    return temp_opt_state


def compute_grads_per_module(in_out_acts):
    grads_per_module = dict()
    for m in in_out_acts:
        grads_per_module[m] = gradparam_per_batch(m, in_out_acts[m])
    return grads_per_module


def compute_grad_losses(
    orig_grads_per_module,
    simple_grads_per_module,
    opt_state,
    temp_opt_state,
    beta1,
    beta2,
    eps,
    divide_mean_loss_by_orig_mean=True,
    eps_mean=1e-2,
):
    grad_match_losses = []
    grad_mean_losses = []
    for m in simple_grads_per_module:
        simple_grads_per_key = simple_grads_per_module[m]
        for key in simple_grads_per_key:
            p = getattr(m, key)

            orig_mean = opt_state["orig_mean"][p] / (1 - beta1 ** opt_state["step"])
            simple_mean = temp_opt_state["simple_mean"][p] / (
                1 - beta1 ** opt_state["step"]
            )
            grad_mean_loss = th.square(orig_mean - simple_mean).sum()
            if divide_mean_loss_by_orig_mean:
                grad_mean_loss = grad_mean_loss / (orig_mean.square().sum() + eps_mean)
            grad_mean_losses.append(grad_mean_loss)

            orig_grads = orig_grads_per_module[m][key]
            simple_grads = simple_grads_per_key[key]
            assert orig_grads.shape == simple_grads.shape
            orig_square = opt_state["orig_square"][p] / (1 - beta2 ** opt_state["step"])
            simple_square = temp_opt_state["simple_square"][p] / (
                1 - beta2 ** opt_state["step"]
            )
            # now similar to cos dist
            renormed_orig_grads = orig_grads / th.sqrt(orig_square.unsqueeze(0) + eps)
            renormed_simple_grads = simple_grads / th.sqrt(
                simple_square.unsqueeze(0) + eps
            )
            grad_match_loss = (
                th.sum(
                    th.square(renormed_orig_grads - renormed_simple_grads), dim=0
                ).mean()
                / 2
            )
            grad_match_losses.append(grad_match_loss)
    return grad_mean_losses, grad_match_losses


def compute_grad_losses_learned_square_fraction(
    orig_grads_per_module,
    simple_grads_per_module,
    opt_state,
    learned_square_alphas,
    beta1,
    beta2,
    eps,
    divide_mean_loss_by_orig_mean=True,
    eps_mean=1e-2,
):
    grad_match_losses = []
    grad_mean_losses = []
    for m in simple_grads_per_module:
        simple_grads_per_key = simple_grads_per_module[m]
        for key in simple_grads_per_key:
            p = getattr(m, key)
            square_fraction = learned_square_alphas[p]
            orig_mean = opt_state["orig_mean"][p] / (1 - beta1 ** opt_state["step"])
            simple_mean = square_fraction * orig_mean
            grad_mean_loss = th.square(orig_mean - simple_mean).sum()
            if divide_mean_loss_by_orig_mean:
                grad_mean_loss = grad_mean_loss / (orig_mean.square().sum() + eps_mean)
            grad_mean_losses.append(grad_mean_loss)

            orig_grads = orig_grads_per_module[m][key]
            simple_grads = simple_grads_per_key[key]
            assert orig_grads.shape == simple_grads.shape
            orig_square = opt_state["orig_square"][p] / (1 - beta2 ** opt_state["step"])
            simple_square = orig_square * square_fraction
            # now similar to cos dist
            renormed_orig_grads = orig_grads / th.sqrt(orig_square.unsqueeze(0) + eps)
            renormed_simple_grads = simple_grads / th.sqrt(
                simple_square.unsqueeze(0) + eps
            )
            grad_match_loss = (
                th.sum(
                    th.square(renormed_orig_grads - renormed_simple_grads), dim=0
                ).mean()
                / 2
            )
            grad_match_losses.append(grad_match_loss)
    return grad_mean_losses, grad_match_losses


def compute_grad_losses_learned_square(
    orig_grads_per_module,
    simple_grads_per_module,
    opt_state,
    learned_square_alphas,
    beta1,
    beta2,
    eps,
    divide_mean_loss_by_orig_mean=True,
    eps_mean=1e-2,
):
    grad_match_losses = []
    grad_mean_losses = []
    for m in simple_grads_per_module:
        simple_grads_per_key = simple_grads_per_module[m]
        for key in simple_grads_per_key:
            p = getattr(m, key)
            orig_square = opt_state["orig_square"][p] / (1 - beta2 ** opt_state["step"])
            square_fraction = th.square(learned_square_alphas[p]) / orig_square
            orig_mean = opt_state["orig_mean"][p] / (1 - beta1 ** opt_state["step"])
            simple_mean = square_fraction * orig_mean
            grad_mean_loss = th.square(orig_mean - simple_mean).sum()
            if divide_mean_loss_by_orig_mean:
                grad_mean_loss = grad_mean_loss / (orig_mean.square().sum() + eps_mean)
            grad_mean_losses.append(grad_mean_loss)

            orig_grads = orig_grads_per_module[m][key]
            simple_grads = simple_grads_per_key[key]
            assert orig_grads.shape == simple_grads.shape
            simple_square = orig_square * square_fraction
            # now similar to cos dist
            renormed_orig_grads = orig_grads / th.sqrt(orig_square.unsqueeze(0) + eps)
            renormed_simple_grads = simple_grads / th.sqrt(
                simple_square.unsqueeze(0) + eps
            )
            grad_match_loss = (
                th.sum(
                    th.square(renormed_orig_grads - renormed_simple_grads), dim=0
                ).mean()
                / 2
            )
            grad_match_losses.append(grad_match_loss)
    return grad_mean_losses, grad_match_losses


def compute_grad_losses_simple_square(
    orig_grads_per_module,
    simple_grads_per_module,
    opt_state,
    temp_opt_state,
    beta1,
    beta2,
    eps,
    divide_mean_loss_by_orig_mean=True,
    eps_mean=1e-2,
):
    grad_match_losses = []
    grad_mean_losses = []
    for m in simple_grads_per_module:
        simple_grads_per_key = simple_grads_per_module[m]
        for key in simple_grads_per_key:
            p = getattr(m, key)
            orig_square = opt_state["orig_square"][p] / (1 - beta2 ** opt_state["step"])
            simple_square = temp_opt_state["simple_square"][p] / (
                1 - beta2 ** opt_state["step"]
            )
            square_fraction = simple_square / orig_square
            orig_mean = opt_state["orig_mean"][p] / (1 - beta1 ** opt_state["step"])
            simple_mean = square_fraction * orig_mean
            grad_mean_loss = th.square(orig_mean - simple_mean).sum()
            if divide_mean_loss_by_orig_mean:
                grad_mean_loss = grad_mean_loss / (orig_mean.square().sum() + eps_mean)
            grad_mean_losses.append(grad_mean_loss)

            orig_grads = orig_grads_per_module[m][key]
            simple_grads = simple_grads_per_key[key]
            assert orig_grads.shape == simple_grads.shape
            # now similar to cos dist
            renormed_orig_grads = orig_grads / th.sqrt(orig_square.unsqueeze(0) + eps)
            renormed_simple_grads = simple_grads / th.sqrt(
                simple_square.unsqueeze(0) + eps
            )
            grad_match_loss = (
                th.sum(
                    th.square(renormed_orig_grads - renormed_simple_grads), dim=0
                ).mean()
                / 2
            )
            grad_match_losses.append(grad_match_loss)
    return grad_mean_losses, grad_match_losses
