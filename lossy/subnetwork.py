from torch import nn
import torch as th
import numpy as np
from functools import partial

class MaskGenerator(nn.Module):
    def __init__(self, model):
        super().__init__()
        # alphas list just so that parameters are registered properly
        alphas_list = nn.ParameterList()
        biases_mask_list = []
        uses_alphas_list = []
        for n, p in model.named_parameters():
            if 'weight' in n:
                param = nn.Parameter(th.zeros_like(p).detach())
                alphas_list.append(param)
                uses_alphas_list.append(True)
            else:
                assert 'bias' in n
                biases_mask_list.append(th.ones_like(p.data, requires_grad=False))
                uses_alphas_list.append(False)

        self.biases_mask_list = biases_mask_list
        self.alphas_list = alphas_list
        self.uses_alphas_list = uses_alphas_list

    def forward(self, with_uni_noise):
        i_bias = 0
        i_alpha = 0
        all_masks = []
        for uses_alphas in self.uses_alphas_list:
            if uses_alphas:
                alpha = self.alphas_list[i_alpha]
                lower_bound = th.sigmoid(alpha)
                if with_uni_noise:
                    noise = th.rand_like(lower_bound)
                    noise = noise * (1 - lower_bound.detach())
                    mask = lower_bound + noise
                else:
                    mask = lower_bound
                i_alpha += 1
            else:
                mask = self.biases_mask_list[i_bias]
                i_bias += 1
            all_masks.append(mask)
        return all_masks


class ParamGenerator(nn.Module):
    def __init__(self, mask_gen):
        super().__init__()
        self.mask_gen = mask_gen

    def forward(self, f_model, with_uni_noise):
        masks = self.mask_gen(with_uni_noise=with_uni_noise)
        f_params = [p * m for p, m in zip(f_model.parameters(), masks)]
        return f_params


class ParamMixMasksGenerator(nn.Module):
    def __init__(self, mask_gen_a, mask_gen_b):
        super().__init__()
        self.mask_gen_a = mask_gen_a
        self.mask_gen_b = mask_gen_b

    def forward(self, f_model,):
        masks_a = self.mask_gen_a(with_uni_noise=False)
        masks_b = self.mask_gen_b(with_uni_noise=False)
        masks = [th.maximum(a,b) for a,b in zip(masks_a, masks_b)]

        f_params = [p * m for p, m in zip(f_model.parameters(), masks)]
        return f_params, masks_a, masks_b


class ClfAndMaskGen(nn.Module):
    def __init__(self, clf, mask_gen):
        super().__init__()
        self.clf = clf
        self.mask_gen = mask_gen

    def forward(self, clf_or_mask_gen, *args, **kwargs):
        if clf_or_mask_gen == 'clf':
            return self.clf(*args, **kwargs)
        else:
            assert clf_or_mask_gen == 'mask_gen'
            return self.mask_gen(*args, **kwargs)


def alpha_l1_mean(masks):
    alpha_l1_sum = th.sum(th.stack([th.sum(m) for m in masks]))
    n_params = np.sum([m.numel() for m in masks])
    alpha_l1_loss = alpha_l1_sum / n_params
    return alpha_l1_loss


class PerExampleParamsNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, per_example_params):
        i_param = 0
        for child in self.net.children():
            if len(list(child.parameters())) == 0:
                x = child(x)
            else:
                assert hasattr(child, 'weight')
                assert hasattr(child, 'bias')
                assert len(x) == len(per_example_params)
                weight_params = [p[i_param] for p in per_example_params]
                bias_params = [p[i_param + 1] for p in per_example_params]
                i_param += 2
                x = th.sum(th.stack(weight_params) * x.unsqueeze(1), dim=-1)
                x = x + th.stack(bias_params)
        return x


class PerExampleMaskParamsNet(nn.Module):
    def __init__(self, net, n_examples):
        super().__init__()
        mask_generators = [MaskGenerator(net) for _ in range(n_examples)]
        self.mask_generators = nn.ModuleList(mask_generators)
        self.net = net

    def forward(self, x, i_example_to_mix):
        if i_example_to_mix == 'get_masks':
            return [mask_gen(with_uni_noise=False) for mask_gen in self.mask_generators]
        elif i_example_to_mix == 'orig_params':
            return self.net(x)
        elif i_example_to_mix is None:
            per_example_params = [params_from_mask(
                mask_gen, self.net, with_uni_noise=False)[1]
                                  for mask_gen in self.mask_generators]
        else:
            assert len(i_example_to_mix) == len(self.mask_generators)
            per_example_params = [
                ParamMixMasksGenerator(
                    self.mask_generators[i_a], self.mask_generators[i_b])(self.net)[0]
                for i_a, i_b in zip(range(len(self.mask_generators)), i_example_to_mix)]
        return PerExampleParamsNet(self.net)(x, per_example_params)


def params_from_mask(mask_gen, net, with_uni_noise=False):
    masks = mask_gen(with_uni_noise=with_uni_noise)
    params = [p * m for p,m in zip(net.parameters(), masks)]
    return masks, params


def get_out_for_gates(hooked_net, module_to_gates, X_clf):
    modules_net = list(hooked_net.modules())
    for module, gates in module_to_gates.items():
        assert module in modules_net
        module.gates_wrapper.gates = gates
    output = hooked_net(X_clf)
    return output


def get_gated_outputs(hooked_net, this_module_to_alphas, X_clf, i_mix):
    module_to_gates_per_example = {}
    for module in this_module_to_alphas[0]:
        all_gates = [th.sigmoid(m_to_a[module]) for m_to_a in this_module_to_alphas]
        module_to_gates_per_example[module] = th.stack(all_gates).unsqueeze(-1).unsqueeze(-1)
    output_per_example_nets = get_out_for_gates(hooked_net, module_to_gates_per_example, X_clf)

    module_to_mixed_gates = {}
    for module, gates in module_to_gates_per_example.items():
        mixed_gates = th.maximum(gates, gates[i_mix])
        module_to_mixed_gates[module] = mixed_gates
    output_mixed_nets = get_out_for_gates(hooked_net, module_to_mixed_gates, X_clf)
    return output_per_example_nets, output_mixed_nets


def create_gates_per_example(module_to_alphas):
    module_to_gates_per_example = {}
    for module in module_to_alphas[0]:
        all_gates = [th.sigmoid(m_to_a[module]) for m_to_a in module_to_alphas]
        module_to_gates_per_example[module] = th.stack(all_gates).unsqueeze(-1).unsqueeze(-1)
    return module_to_gates_per_example


def get_gated_outputs_mixed(hooked_net, this_module_to_alphas, mix_module_to_alphas, X_clf,):
    module_to_gates_per_example = create_gates_per_example(this_module_to_alphas)
    module_to_gates_mix_per_example = create_gates_per_example(mix_module_to_alphas)
    output_per_example_nets = get_out_for_gates(hooked_net, module_to_gates_per_example, X_clf)

    module_to_mixed_gates = {}
    for (module, gates), (module2, gates2) in zip(
            module_to_gates_per_example.items(), module_to_gates_mix_per_example.items()):
        assert module == module2
        mixed_gates = th.maximum(gates, gates2)
        module_to_mixed_gates[module] = mixed_gates
    output_mixed_nets = get_out_for_gates(hooked_net, module_to_mixed_gates, X_clf)
    return output_per_example_nets, output_mixed_nets


def scale_acts_by_gates(module, input, output, gates):
    #print("out", output[0,0,:2,:2])
    #print("gates", gates[0,0,:2,:2])
    return output * gates


def forward_gated(net, name_to_gates, X,):
    handles = []
    for name, module in net.named_modules():
        if name in name_to_gates:
            gates = name_to_gates[name]
            handle = module.register_forward_hook(partial(scale_acts_by_gates, gates=gates))
            handles.append(handle)
    try:
        out = net(X)
    finally:
        for h in handles:
            h.remove()
    return out


def l1_mean_gates(alphas):
    l1_sum = sum([th.sum(th.sigmoid(a)) for a in alphas])
    n_gates = sum([a.numel() for a in alphas])
    l1_mean = l1_sum/n_gates
    return l1_mean


def round_with_gradient(x):
    return x + (x.round() - x).detach()


def identity(x):
    return x


class GateForward(nn.Module):
    def __init__(self, name_to_alpha, ):
        super().__init__()
        self.alphas = nn.ParameterList(name_to_alpha.values())
        # have to do like this to ensure alphas are correctly seen by higher
        self.names = list(name_to_alpha.keys())

    def forward(self, net, X, do_round, mix_alpha=None, pass_through_grad=True, just_return_alphas=False):
        if just_return_alphas:
            return self.alphas
        round_fn = [identity, round_with_gradient, ][do_round]
        name_to_gates = {name: th.sigmoid(alpha).unsqueeze(-1).unsqueeze(-1)
                         for name, alpha in zip(self.names, self.alphas)}
        if mix_alpha is not None:
            name_to_other_gates = {name: round_fn(th.sigmoid(alpha).unsqueeze(-1).unsqueeze(-1))
                                   for name, alpha in mix_alpha.items()}
            if pass_through_grad:
                name_to_gates_mixed = {name: (
                        round_fn(gate - gate.detach() + th.maximum(gate, other_gate).detach()))
                    for (name, gate), (name2, other_gate)
                    in zip(name_to_gates.items(), name_to_other_gates.items())}
            else:
                name_to_gates_mixed = {name: round_fn(th.maximum(gate, other_gate.detach()))
                    for (name, gate), (name2, other_gate)
                    in zip(name_to_gates.items(), name_to_other_gates.items())}

            name_to_gates = name_to_gates_mixed
        out_gated = forward_gated(net, name_to_gates, X)
        return out_gated


def compute_kl_divs_gated(gater, net, X, out_orig, do_round, mix_alpha, pass_through_grad, n_mix_samples):
    from lossy.experiments.run import symmetric_kl_divergence
    # pass_through_grad no effect here at the moment
    out_gated = gater(net, X, do_round=do_round, pass_through_grad=pass_through_grad)
    kl_div = symmetric_kl_divergence(out_gated, out_orig)
    # sample as many random permutations as desired
    kl_divs_mixed = []
    for _ in range(n_mix_samples):
        i_mix = th.randperm(len(list(mix_alpha.values())[0]))[:len(X)]
        this_mix_alpha = mix_alphas(mix_alpha, i_mix)
        this_kl_div_mixed = compute_kl_divs_gated_mixed(
            gater, net, X, out_orig, do_round=do_round, mix_alpha=this_mix_alpha,
            pass_through_grad=pass_through_grad)
        kl_divs_mixed.append(this_kl_div_mixed)
    kl_div_mixed = th.mean(th.stack(kl_divs_mixed))
    return kl_div, kl_div_mixed


def compute_kl_divs_gated_mixed(gater, net, X, out_orig, do_round, mix_alpha, pass_through_grad):
    from lossy.experiments.run import symmetric_kl_divergence
    out_gated_mixed_ = gater(net, X, do_round=do_round, mix_alpha=mix_alpha, pass_through_grad=pass_through_grad)
    kl_div_mixed = symmetric_kl_divergence(out_gated_mixed_, out_orig)
    return kl_div_mixed


def mix_alphas(name_to_alpha, i_mix):
    mixed_name_to_alpha = {}
    for name, alpha in name_to_alpha.items():
        mixed_name_to_alpha[name] = alpha[i_mix]
    return mixed_name_to_alpha
