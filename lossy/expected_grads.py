import torch as th
import numpy as np


def get_contrib_per_pixel_from_module(
    clf,
    wanted_module,
    this_X,
    simple_X,
    batch_size,
    wanted_y,
    retain_graph=True,
    create_graph=False,
    detach_simple_acts=True,
    on_output=False,
):
    X = th.cat((this_X, simple_X))
    orig_simple_mixed_acts = []

    def create_mixed_inputs(module, input, output):
        if on_output:
            orig_acts, simple_acts = th.split(
                output, [len(this_X), len(simple_X)]
            )  # first split input into ref and baseline
        else:
            orig_acts, simple_acts = th.split(
                input[0], [len(this_X), len(simple_X)]
            )  # first split input into ref and baseline
        baseline_acts = simple_acts.unsqueeze(1)
        if detach_simple_acts:
            baseline_acts = baseline_acts.detach()
        alpha = th.linspace(0, 1, batch_size, device="cuda")
        expanded_alpha = alpha.view(1, -1, *((1,) * (baseline_acts.ndim - 2)))
        mixed_acts = (
            expanded_alpha * orig_acts.unsqueeze(1)
            + (1 - expanded_alpha) * baseline_acts
        )
        if detach_simple_acts:
            mixed_acts = mixed_acts.detach()
        mixed_acts = mixed_acts.requires_grad_(True)
        # mixed_acts = mixed_acts.requires_grad_(True)
        mixed_acts_flat = mixed_acts.reshape(
            np.prod(mixed_acts.shape[:2]), *mixed_acts.shape[2:]
        )
        orig_simple_mixed_acts.append(orig_acts)
        orig_simple_mixed_acts.append(simple_acts)
        orig_simple_mixed_acts.append(mixed_acts)
        handle.remove()
        if on_output:
            return mixed_acts_flat
        else:
            return module(mixed_acts_flat)

    handle = wanted_module.register_forward_hook(create_mixed_inputs)

    try:
        out = clf(X)
        out_reshaped = th.log_softmax(out, dim=1).reshape(len(this_X), batch_size, -1)
        out_summed = out_reshaped.sum(dim=1)
        loss = out_summed.gather(1, wanted_y.unsqueeze(1)).sum()  # loss_fn(out_summed)
        orig_acts, simple_acts, mixed_acts = orig_simple_mixed_acts
        grads = th.autograd.grad(loss, mixed_acts, retain_graph=retain_graph,
                                 create_graph=create_graph)
    finally:
        handle.remove()

    avg_grad = grads[0].mean(dim=1)
    if detach_simple_acts:
        avg_grad = avg_grad.detach()

    contrib_per_pixel = avg_grad * (simple_acts - orig_acts.detach())
    return contrib_per_pixel


def forward_with_masked_activations(
    net,
    X,
    masks_per_module,
):
    def mask_activations(module, input, output):
        return output * masks_per_module[module].view_as(output)

    handles = []
    for module in masks_per_module:
        handle = module.register_forward_hook(mask_activations)
        handles.append(handle)
    try:
        out = net(X)
    finally:
        for h in handles:
            h.remove()
    return out


def get_all_contribs(
    clf,
    X,
    wanted_y,
    wanted_modules,
    act_shapes_per_module,
    batch_size,
    create_graph=False,
):
    masks_per_module = {}
    for m in wanted_modules:
        mask = th.ones(
            len(X),
            batch_size,
            *act_shapes_per_module[m][1:],
            requires_grad=True,
            device="cuda"
        )
        alpha = th.linspace(0, 1, batch_size, device="cuda")
        expanded_alpha = alpha.view(1, -1, *((1,) * (mask.ndim - 2)))
        mask.data[:] = (mask * expanded_alpha).data[:]
        masks_per_module[m] = mask

    flat_X_for_masking = (
        X.unsqueeze(1).repeat(1, batch_size, 1, 1, 1).view(-1, *X.shape[1:])
    )
    out = forward_with_masked_activations(clf, flat_X_for_masking, masks_per_module)
    out_reshaped = th.log_softmax(out, dim=1).reshape(len(X), len(alpha), -1)
    out_summed = out_reshaped.sum(dim=1)
    loss = out_summed.gather(1, wanted_y.unsqueeze(1)).sum()  # loss_fn(out_summed)
    grads = th.autograd.grad(
        loss, [m for m in masks_per_module.values()], create_graph=create_graph
    )

    avg_grads = [g.mean(dim=1) for g in grads]

    contribs = th.cat([th.flatten(g, 1) for g in avg_grads], dim=1)
    return contribs
