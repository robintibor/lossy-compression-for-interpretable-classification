from copy import copy
import torch as th
import numpy as np
from einops import rearrange
from functools import partial
from torch import nn


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
    net, X, loss_fn, wanted_modules=None, **backward_kwargs
):
    if wanted_modules is None:
        wanted_modules = net.modules()

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
        assert "in_act" not in module_to_vals[module]
        module_to_vals[module]["in_act"] = input
        assert "out_act" not in module_to_vals[module]
        if hasattr(output, 'register_hook'):
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



def get_in_out_activations_per_module(net, X, wanted_modules=None, **backward_kwargs):
    if wanted_modules is None:
        wanted_modules = net.modules()

    module_to_vals = {m: {} for m in wanted_modules}

    def append_activations(module, input, output):
        assert "in_act" not in module_to_vals[module]
        module_to_vals[module]["in_act"] = input
        assert "out_act" not in module_to_vals[module]
        if hasattr(output, 'register_hook'):
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


def conv_batch_grad(module, in_act, out_grad):
    assert np.all(np.array(module.kernel_size) // 2 == np.array(module.padding))
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
    weight_grad = conv_weight_grad_loop(
        module, in_act, out_grad
    )  # conv_weight_grad_backpack(module, in_act, out_grad)

    bias_batch_grad = out_grad.sum(dim=(-2, -1))
    return weight_grad, bias_batch_grad


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
    grads = []
    for i_ex in range(len(in_act)):
        grad = conv_weight_grad(
            module, in_act[i_ex : i_ex + 1], out_grad[i_ex : i_ex + 1]
        )
        grads.append(grad)
    return th.stack(grads)


def conv_weight_grad(module, in_act, out_grad):
    recreated_weight_grad = th.nn.functional.conv2d(
        in_act.transpose(0, 1),
        out_grad.transpose(0, 1),
        padding=module.padding,
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


def grad_out_act_act(m, x_m_vals, ref_m_vals):
    return [
        (
           x_a * x_g, r_a * r_g
        )
        for x_a, x_g, r_a, r_g in zip(
            x_m_vals["out_act"], x_m_vals["out_grad"],
            ref_m_vals["out_act"], ref_m_vals["out_grad"],

        )
    ]


def grad_act_relued(m, x_m_vals, ref_m_vals):
    vals = []
    for x_a, x_g, r_a, r_g in zip(
            x_m_vals["out_act"], x_m_vals["out_grad"],
            ref_m_vals["out_act"], ref_m_vals["out_grad"],

    ):
        mask = (r_a * r_g) > 0
        vals.append((mask * x_g), (mask * r_g))
    return vals


def grad_in_act_act(m, x_m_vals, ref_m_vals):
    return [
        (
           x_a * x_g, r_a * r_g
        )
        for x_a, x_g, r_a, r_g in zip(
            x_m_vals["in_act"], x_m_vals["in_grad"],
            ref_m_vals["in_act"], ref_m_vals["in_grad"],
        )
    ]


def grad_in_act_act_same_sign(m , x_m_vals, ref_m_vals):
    vals = []
    for x_a, x_g, r_a, r_g in zip(
        x_m_vals["in_act"], x_m_vals["in_grad"],
        ref_m_vals["in_act"], ref_m_vals["in_grad"],
    ):
        mask_ref = (r_a * r_g) > 0
        mask = (mask_ref * (x_g.sign() == r_g.sign()))

        x_val = (r_g * x_a)  #clip_min_max(xg, rg) *
        ref_val = (mask * r_g * r_a)
        vals.append((x_val, ref_val))
    return vals


def grad_in_act_act_same_sign_masked(m , x_m_vals, ref_m_vals):
    vals = []
    for x_a, x_g, r_a, r_g in zip(
        x_m_vals["in_act"], x_m_vals["in_grad"],
        ref_m_vals["in_act"], ref_m_vals["in_grad"],
    ):
        mask_ref = (r_a * r_g) > 0
        mask = (mask_ref * (x_g.sign() == r_g.sign()))

        x_val = (r_g * x_a * mask)  #clip_min_max(xg, rg) *
        ref_val = (mask * r_g * r_a)
        vals.append((x_val, ref_val))
    return vals


def gradparam_per_batch(m, vals):
    assert m.__class__.__name__ in ["BatchNorm2d", "Conv2d", "Linear"]
    grad_fn = {
        "BatchNorm2d": bnorm_batch_grad,
        "Conv2d": conv_batch_grad,
        "Linear": linear_batch_grad,
    }[m.__class__.__name__]

    weight_batch_grad, bias_batch_grad = grad_fn(
        m, vals["in_act"][0], vals["out_grad"][0]
    )
    grads = {}
    if m.weight is not None:
        grads["weight"] = weight_batch_grad
    if m.bias is not None:
        grads["bias"] = bias_batch_grad
    return grads


def gradparam(m, x_m_vals, ref_m_vals):
    assert m.__class__.__name__ in ["BatchNorm2d", "Conv2d", "Linear"]
    x_grads = gradparam_per_batch(m, x_m_vals)
    ref_grads = gradparam_per_batch(m, ref_m_vals)
    grad_tuples = [(x_grads[key], ref_grads[key]) for key in x_grads]
    return grad_tuples



def gradparam_relued(m, x_m_vals, ref_m_vals):
    assert m.__class__.__name__ in ["BatchNorm2d", "Conv2d", "Linear"]
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


def gradparam_param(m, x_m_vals, ref_m_vals):
    assert m.__class__.__name__ in ["BatchNorm2d", "Conv2d", "Linear"]
    x_grads = gradparam_per_batch(m, x_m_vals)
    ref_grads = gradparam_per_batch(m, ref_m_vals)

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
            (th.where(ref_val > 0, th.minimum(x_val, ref_val), x_val), ref_val) for x_val, ref_val in x_and_ref_vals
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


def compute_dist(dist_fn, val_fn,
                 X_act_grads, ref_act_grads, ):
    dists = []
    for m in X_act_grads:
        vals = val_fn(m, X_act_grads[m], ref_act_grads[m])
        for val_x, val_ref in vals:
            dist = dist_fn(th.flatten(val_x, start_dim=1), th.flatten(val_ref, start_dim=1))
            dists.append(dist)
    return dists


def compute_multiple_dists(dist_fns, val_fn, X_act_grads, ref_act_grads):
    dists_per_fn = {d_fn: [] for d_fn in dist_fns}
    for m in X_act_grads:
        vals = val_fn(m, X_act_grads[m], ref_act_grads[m])
        for val_x, val_ref in vals:
            for dist_fn in dist_fns:
                dist = dist_fn(th.flatten(val_x, start_dim=1), th.flatten(val_ref, start_dim=1))
                dists_per_fn[dist_fn].append(dist)
    return dists_per_fn


def cosine_distance(*args, **kwargs):
    return 1 - th.nn.functional.cosine_similarity(*args, **kwargs)


def sse_loss(x, *args, **kwargs):
    return th.nn.functional.mse_loss(x, *args, **kwargs, reduction='none').sum(
        dim=tuple(range(1, len(x.shape))))


def mse_loss(x, *args, **kwargs):
    return th.nn.functional.mse_loss(x, *args, **kwargs, reduction='none').mean(
        dim=tuple(range(1, len(x.shape))))


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
    mses = th.nn.functional.mse_loss(val_x, val_ref, *args, **kwargs, reduction='none').mean(
        dim=tuple(range(1, len(val_x.shape))))
    return mses / th.mean(val_ref * val_ref, dim=tuple(range(1, len(val_x.shape))))


def detach_acts_grads(acts_grads):
    for m in acts_grads:
        for key in acts_grads[m]:
            acts_grads[m][key] = [a.detach() if hasattr(a, 'detach') else a for a in acts_grads[m][key]]
    return acts_grads


def filter_act_grads(act_grads, wanted_keys):
    filtered = {}
    for m in act_grads:
        if all([key in act_grads[m] for key in wanted_keys]):
            filtered[m] = act_grads[m]
    return filtered
