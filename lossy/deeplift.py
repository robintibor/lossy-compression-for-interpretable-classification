import torch
# from https://github.com/pytorch/captum/blob/945c582cc0b08885c4e2bfecb020abdfac0122f3/captum/_utils/common.py#L718C1-L766C6
import typing
from enum import Enum
from functools import reduce
from inspect import signature
from typing import Any, Callable, cast, Dict, List, overload, Tuple, Union
from torch import nn
import torch as th
from functools import partial


def _register_backward_hook(
    module: nn.Module, hook: Callable, attr_obj: Any
) -> List[torch.utils.hooks.RemovableHandle]:
    grad_out: Dict[torch.device, torch.Tensor] = {}

    def forward_hook(
        module: nn.Module,
        inp: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        out: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> None:
        nonlocal grad_out
        grad_out = {}

        def output_tensor_hook(output_grad: torch.Tensor) -> None:
            grad_out[output_grad.device] = output_grad

        if isinstance(out, tuple):
            assert (
                len(out) == 1
            ), "Backward hooks not supported for module with >1 output"
            out[0].register_hook(output_tensor_hook)
        else:
            out.register_hook(output_tensor_hook)

    def pre_hook(module, inp):
        def input_tensor_hook(input_grad: torch.Tensor):
            if len(grad_out) == 0:
                return
            hook_out = hook(module, input_grad, grad_out[input_grad.device])

            if hook_out is not None:
                return hook_out[0] if isinstance(hook_out, tuple) else hook_out

        if isinstance(inp, tuple):
            assert (
                len(inp) == 1
            ), "Backward hooks not supported for module with >1 input"
            inp[0].register_hook(input_tensor_hook)
            return inp[0].clone()
        else:
            inp.register_hook(input_tensor_hook)
            return inp.clone()

    return [
        module.register_forward_pre_hook(pre_hook),
        module.register_forward_hook(forward_hook),
    ]


def compute_deeplift_grad(net, X, ref_acts, loss_fn, wanted_modules_grads, mods_to_modify, **backward_kwargs):
    # probably does not work with nn.DataParallel for now
    handles_backward = []

    def remember_grads(grad_input, ref_in_act, ref_out_act, this_in_act, this_out_act):
        grad_input.ref_in_act = ref_in_act
        grad_input.ref_out_act = ref_out_act
        grad_input.this_in_act = this_in_act
        grad_input.this_out_act = this_out_act
        return grad_input

    def deeplift_modify_grad(module, grad_input, grad_output, save_in_grad):
        ref_in_act = grad_input.ref_in_act
        ref_out_act = grad_input.ref_out_act
        this_in_act = grad_input.this_in_act
        this_out_act = grad_input.this_out_act
        del grad_input.ref_in_act
        del grad_input.ref_out_act
        del grad_input.this_in_act
        del grad_input.this_out_act
        in_diffs = this_in_act - ref_in_act
        mask = th.abs(in_diffs) > 1e-7
        out_diffs = this_out_act - ref_out_act
        in_diffs = in_diffs + (~mask)  # prevent division by zero
        fraction = out_diffs / in_diffs
        new_in_grad_from_diff = fraction * grad_output
        new_grad = (new_in_grad_from_diff * mask) + (grad_input * ~mask)
        if save_in_grad:
            module_to_vals[module]['in_grad'] = new_grad
        return new_grad

    def remember_activations(module, input, output):
        ref_in_act, ref_out_act = ref_acts.pop(0)
        assert len(ref_in_act) == 1
        ref_in_act = ref_in_act[0]
        assert len(input) == 1
        this_in_act = input[0]

        handle = input[0].register_hook(partial(remember_grads, ref_in_act=ref_in_act, ref_out_act=ref_out_act,
                                                this_in_act=this_in_act, this_out_act=output))
        handles_backward.append(handle)

    module_to_vals = {}

    def append_activations(module, input, output, add_grad_input):
        module_to_vals[module]["in_act"] = input
        module_to_vals[module]["out_act"] = output
        if add_grad_input:
            for a_input in input:
                # see https://github.com/pytorch/pytorch/issues/25723 for why like this
                handle = a_input.register_hook(
                    partial(
                        append_grad_inputs,
                        module,
                    )
                )
                handles_backward.append(handle)
        handle = output.register_hook(
            partial(
                append_grad_outputs,
                module,
            )
        )
        handles_backward.append(handle)

    def append_grad_inputs(module, grad_input, ):
        assert grad_input is not None
        module_to_vals[module]['in_grad'] = grad_input

    def append_grad_outputs(module, grad_output, ):
        assert grad_output is not None
        module_to_vals[module]["out_grad"] = grad_output

    modules_to_handles = {}
    for module in mods_to_modify:
        handle = module.register_forward_hook(remember_activations)
        modules_to_handles[module] = handle
        handles = _register_backward_hook(module, partial(
            deeplift_modify_grad, save_in_grad=module in wanted_modules_grads), None)
        for h in handles:
            handles_backward.append(h)
    for module in wanted_modules_grads:
        module_to_vals[module] = {}
        handle = module.register_forward_hook(
            partial(append_activations, add_grad_input=module not in mods_to_modify))
        handles_backward.append(handle)  # hack fow now, try make clearer
    try:
        out = net(X)
        loss = loss_fn(out)
        loss.backward(**backward_kwargs)
    finally:
        for m in modules_to_handles:
            modules_to_handles[m].remove()
        for h in handles_backward:
            h.remove()
    return module_to_vals


class DeepLiftHook():
    def __init__(self, module, module_to_vals):
        self.grad_output = None
        self.grad_input = None
        self.this_in_act = None
        self.this_out_act = None
        self.ref_in_act = None
        self.ref_out_act = None
        self.module = module
        self.module_to_vals = module_to_vals

    def remember_acts(self, this_in_act, this_out_act, ref_in_act, ref_out_act):
        #print("remember acts for", self.module, "with hook id", id(self), "mod id", id(self.module))
        self.ref_in_act = ref_in_act
        self.ref_out_act = ref_out_act
        self.this_in_act = this_in_act
        self.this_out_act = this_out_act

    def remember_grad_out(self, grad_output):
        self.grad_output = grad_output

    def deeplift_modify_grad(self, grad_input):
        #print("mod grad for", self.module, "with hook id", id(self), "mod id", id(self.module))
        grad_output = self.grad_output
        this_in_act = self.this_in_act
        this_out_act = self.this_out_act
        ref_in_act = self.ref_in_act
        ref_out_act = self.ref_out_act
        in_diffs = this_in_act - ref_in_act
        mask = th.abs(in_diffs) > 1e-7
        out_diffs = this_out_act - ref_out_act
        in_diffs = in_diffs + (~mask)  # prevent division by zero
        fraction = out_diffs / in_diffs
        new_in_grad_from_diff = fraction * grad_output
        new_grad = (new_in_grad_from_diff * mask) + (grad_input * ~mask)
        if self.module in self.module_to_vals:
            self.module_to_vals[self.module]['in_grad'] = new_grad
        self.grad_output = None
        self.grad_input = None
        self.this_in_act = None
        self.this_out_act = None
        self.ref_in_act = None
        self.ref_out_act = None
        return new_grad


def compute_deeplift_grad_3(net, X, ref_acts, loss_fn, wanted_modules_grads, mods_to_modify):
    # probably does not work with nn.DataParallel for now
    handles_backward = []


    def remember_acts_for_deeplift(module, input, output,):
        ref_in_act, ref_out_act = ref_acts.pop(0)
        assert len(ref_in_act) == 1
        ref_in_act = ref_in_act[0]
        assert len(input) == 1
        this_in_act = input[0]
        deeplift_hook = DeepLiftHook(module, module_to_vals)
        deeplift_hook.remember_acts(this_in_act, output, ref_in_act, ref_out_act)
        handle = output.register_hook(deeplift_hook.remember_grad_out)
        handles_backward.append(handle)
        handle = this_in_act.register_hook(deeplift_hook.deeplift_modify_grad)
        handles_backward.append(handle)

    module_to_vals = {}

    def append_activations(module, input, output, add_grad_input):
        module_to_vals[module]["in_act"] = input
        module_to_vals[module]["out_act"] = output
        if add_grad_input:
            for a_input in input:
                # see https://github.com/pytorch/pytorch/issues/25723 for why like this
                handle = a_input.register_hook(
                    partial(
                        append_grad_inputs,
                        module,
                    )
                )
                handles_backward.append(handle)
        handle = output.register_hook(
            partial(
                append_grad_outputs,
                module,
            )
        )
        handles_backward.append(handle)

    def append_grad_inputs(module, grad_input, ):
        assert grad_input is not None
        module_to_vals[module]['in_grad'] = grad_input

    def append_grad_outputs(module, grad_output, ):
        assert grad_output is not None
        module_to_vals[module]["out_grad"] = grad_output

    modules_to_handles = {}
    for module in mods_to_modify:
        deeplift_hook = DeepLiftHook(module, module_to_vals)
        handle = module.register_forward_hook(remember_acts_for_deeplift)
        modules_to_handles[module] = handle
    for module in wanted_modules_grads:
        module_to_vals[module] = {}
        handle = module.register_forward_hook(
            partial(append_activations, add_grad_input=module not in mods_to_modify))
        handles_backward.append(handle)  # hack fow now, try make clearer
    try:
        out = net(X)
        loss = loss_fn(out)
        loss.backward()
    finally:
        for m in modules_to_handles:
            modules_to_handles[m].remove()
        for h in handles_backward:
            h.remove()
    return module_to_vals


