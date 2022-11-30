import torch
import torch as th
from torch.optim.optimizer import Optimizer, required
from lossy.util import np_to_th, th_to_np
try:
    import higher
except ModuleNotFoundError:
    pass
import logging
log = logging.getLogger(__name__)


class PercentileGradClip(Optimizer):
    def __init__(self, base_optim, percentile, n_history):
        # mostly forward calls to base optim
        # https://stackoverflow.com/a/1445289
        self.__dict__ = base_optim.__dict__
        self.base_optim = base_optim
        self.percentile = percentile
        self.n_history = n_history
        self.grad_norms = []
        

    def step(self):
        grad_norm = th.stack(
            [th.norm(p.grad,p=2)
             for g in self.base_optim.param_groups for p in g['params']]).mean().item()
        self.grad_norms.append(grad_norm)
        if len(self.grad_norms) > self.n_history:
            self.grad_norms = self.grad_norms[1:]
        max_grad_norm = np.percentile(self.grad_norms, self.percentile,
                                     interpolation='lower')
        
        if grad_norm > max_grad_norm:
            factor = max_grad_norm / grad_norm
            for g in self.base_optim.param_groups:
                for p in g['params']:
                    p.grad.data.multiply_(factor)
            
        self.base_optim.step()
        
        

def grads_all_finite(optimizer):
    for g in optimizer.param_groups:
        for p in g['params']:
            if p.grad is None:
                log.warning("Gradient was none on check of finite grads")
            elif not th.all(th.isfinite(p.grad)).item():
                return False
    return True


def set_grads_to_none(params):
    for p in params:
        assert hasattr(p, 'grad')
        p.grad = None
        if hasattr(p, 'grad_batch'):
            p.grad_batch = None


def preprocess_for_clf(x):
    mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
    }

    std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
    }
    th_mean = np_to_th(mean['cifar10'], device=x.device, dtype=np.float32).unsqueeze(
        0).unsqueeze(-1).unsqueeze(-1)
    th_std = np_to_th(std['cifar10'], device=x.device, dtype=np.float32).unsqueeze(
        0).unsqueeze(-1).unsqueeze(-1)
    # if it was greyscale before, keep greyscale
    was_grey_scale = x.shape[1] == 1
    x = (x - th_mean) / th_std
    if was_grey_scale:
        x = th.mean(x, dim=1, keepdim=True)
    return x


def merge_two_opts(opt_a, opt_b):
    assert len(opt_a.param_groups) == 1
    assert len(opt_b.param_groups) == 1
    assert opt_a.__class__ is opt_b.__class__
    opt_merged = opt_a.__class__(opt_a.param_groups + opt_b.param_groups)

    param_groups = opt_merged.state_dict()["param_groups"]

    # This assumes only two param groups,
    # first from opt a,
    # second from opt b
    old_to_new_param_id = dict(
        zip(
            opt_b.state_dict()["state"].keys(),
            opt_merged.state_dict()["param_groups"][1]["params"],
        )
    )

    state = opt_a.state_dict()["state"]
    state_b = opt_b.state_dict()["state"]

    for old_id, new_id in old_to_new_param_id.items():
        state[new_id] = state_b[old_id]
    opt_merged.load_state_dict(dict(param_groups=param_groups, state=state))
    return opt_merged


def cross_ent_loss(clf, X, y):
    clf_out = clf(X)
    cross_ent = th.nn.functional.cross_entropy(clf_out, y)
    return cross_ent


def preproc_and_to_clf(preproc, X):
    X_opt = preproc(X)
    X_opt_clf = preprocess_for_clf(X_opt)
    return X_opt_clf


def higher_loop(
    clf, optim_clf, data, inner_transform_fn, inner_loss_fn, copy_initial_weights
):
    X_opt_clfs = []
    with higher.innerloop_ctx(
        clf, optim_clf, copy_initial_weights=copy_initial_weights
    ) as (func_clf, func_opt):
        for X, y in data:
            X_opt_clf = inner_transform_fn(X=X)
            inner_loss = inner_loss_fn(func_clf, X_opt_clf, y)
            func_opt.step(inner_loss)
            X_opt_clfs.append(X_opt_clf)
    return func_clf, func_opt, X_opt_clfs, X, y


def higher_loss(
    clf,
    optim_clf,
    data,
    inner_transform_fn,
    inner_loss_fn,
    outer_loss_fn,
    copy_initial_weights,
    copy_final_weights,
    only_eval_last,
):
    if copy_final_weights:
        assert not copy_initial_weights, "not tested this combination"

    func_clf, func_opt, X_opts, last_X, last_y = higher_loop(
        clf,
        optim_clf,
        data,
        inner_transform_fn=inner_transform_fn,
        inner_loss_fn=inner_loss_fn,
        copy_initial_weights=copy_initial_weights,
    )

    outer_losses = []
    if only_eval_last:
        last_X_opt = X_opts[-1]
        outer_loss = outer_loss_fn(func_clf, last_X_opt, last_X, last_y)
        loss = th.stack(outer_loss)
    else:
        for (X, y), X_opt in zip(data, X_opts):
            outer_loss = outer_loss_fn(func_clf, X_opt_clf=X_opt, X_orig_clf=X, y=y)
            outer_losses.append(th.stack(outer_loss))
        loss = th.mean(th.stack(outer_losses, dim=0), dim=0)

    if copy_final_weights:
        for orig_p, func_p in zip(clf.parameters(), func_clf.parameters()):
            orig_p.data = func_p.data.detach()
            # see also https://github.com/facebookresearch/higher/issues/10#issuecomment-574695558 for alternative
        assert len(func_opt.state) == 1
        state_dict_state = {}
        for p_clf, (key, state) in zip(clf.parameters(), func_opt.state[0].items()):
            state_dict_state[p_clf] = {
                k: v if not hasattr(v, "detach") else v.detach()
                for k, v in state.items()
            }
        assert len(optim_clf.param_groups) == 1
        optim_clf.param_groups[0]["params"] = list(clf.parameters())
        state_dict = dict(state=state_dict_state, param_groups=optim_clf.param_groups)
        optim_clf.load_state_dict(state_dict)
    return loss, func_clf


def higher_loss_old(
    clf,
    optim_clf,
    data,
    inner_transform_fn,
    inner_loss_fn,
    outer_loss_fn,
    copy_initial_weights,
    copy_final_weights,
    only_eval_last,
):
    if copy_final_weights:
        assert not copy_initial_weights, "not tested this combination"

    func_clf, func_opt, X_opt_clfs, last_X, last_y = higher_loop(
        clf,
        optim_clf,
        data,
        inner_transform_fn=inner_transform_fn,
        inner_loss_fn=inner_loss_fn,
        copy_initial_weights=copy_initial_weights,
    )

    outer_losses = []
    if only_eval_last:
        X_clf = preprocess_for_clf(last_X)
        X_opt_clf = X_opt_clfs[-1]
        outer_loss = outer_loss_fn(func_clf, X_opt_clf, X_clf, last_y)
        loss = th.stack(outer_loss)
    else:
        for (X, y), X_opt_clf in zip(data, X_opt_clfs):
            X_clf = preprocess_for_clf(X)
            outer_loss = outer_loss_fn(
                func_clf, X_opt_clf=X_opt_clf, X_orig_clf=X_clf, y=y
            )
            outer_losses.append(th.stack(outer_loss))
        loss = th.mean(th.stack(outer_losses, dim=0), dim=0)

    if copy_final_weights:
        for orig_p, func_p in zip(clf.parameters(), func_clf.parameters()):
            orig_p.data = func_p.data.detach()
            # see also https://github.com/facebookresearch/higher/issues/10#issuecomment-574695558 for alternative
        assert len(func_opt.state) == 1
        state_dict_state = {}
        for p_clf, (key, state) in zip(clf.parameters(), func_opt.state[0].items()):
            state_dict_state[p_clf] = {
                k: v if not hasattr(v, "detach") else v.detach()
                for k, v in state.items()
            }
        assert len(optim_clf.param_groups) == 1
        optim_clf.param_groups[0]["params"] = list(clf.parameters())
        state_dict = dict(state=state_dict_state, param_groups=optim_clf.param_groups)
        optim_clf.load_state_dict(state_dict)
    return loss, func_clf


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError("Wrong input dimensions")

    return torch.sum(x ** 2, dim=dim, keepdim=keepdim) ** 0.5


class SGD_AGC(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__
    AGC from NFNets: https://arxiv.org/abs/2102.06171.pdf.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        dampening (float, optional): dampening for momentum (default: 0.01)
        clipping (float, optional): clipping value (default: 1e-3)
        eps (float, optional): eps (default: 1e-3)
    Example:
        >>> optimizer = torch.optim.SGD_AGC(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    .. note::
        The implementation has been adapted from the PyTorch framework and the official
        NF-Nets paper.
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        clipping=1e-2,
        eps=1e-3,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            clipping=clipping,
            eps=eps,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_AGC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_AGC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_norm = torch.max(
                    unitwise_norm(p.detach()), torch.tensor(group["eps"]).to(p.device)
                )
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * group["clipping"]

                trigger = grad_norm > max_norm

                clipped_grad = p.grad * (
                    max_norm
                    / torch.max(grad_norm, torch.tensor(1e-6).to(grad_norm.device))
                )
                p.grad.detach().copy_(torch.where(trigger, clipped_grad, p.grad))

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
