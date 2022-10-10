import torch as th

def grad_normed_loss(unnormed_loss_fn, eps=1e-10):
    def normed_loss_fn(out):
        out_with_grad = out.detach().requires_grad_(True)
        unnormed_loss = unnormed_loss_fn(out_with_grad).mean()
        grads = th.autograd.grad(unnormed_loss, out_with_grad)[0]
        factors = 1 / (th.norm(grads, p=1, dim=1,) + eps)
        losses = unnormed_loss_fn(out)
        assert factors.shape == losses.shape
        loss = th.mean(losses * factors)
        return loss
    return normed_loss_fn

def kl_divergence(clf_out_a, clf_out_b, reduction='mean'):
    assert clf_out_a.shape == clf_out_b.shape
    kl_divs_per_example = th.sum(
                th.nn.functional.softmax(clf_out_a, dim=1) *
                (th.nn.functional.log_softmax(clf_out_a, dim=1) -
                 th.nn.functional.log_softmax(clf_out_b, dim=1)), dim=1)
    if reduction == 'mean':
        kl_div = th.mean(kl_divs_per_example)
    elif reduction == 'sum':
        kl_div = th.sum(kl_divs_per_example)
    else:
        assert reduction is None or reduction == 'none'
        kl_div = kl_divs_per_example
    return kl_div


def symmetric_kl_divergence(clf_out_a, clf_out_b):
    kl_div_a_b = kl_divergence(clf_out_a, clf_out_b)
    kl_div_b_a = kl_divergence(clf_out_b, clf_out_a)
    return (kl_div_a_b + kl_div_b_a) / 2


def soft_cross_entropy_from_logits(logits, target, reduction='mean'):
    log_preds = th.log_softmax(logits, dim=1)
    cents_per_example = -(target * log_preds).sum(dim=1)
    if reduction == 'mean':
        cents = th.mean(cents_per_example)
    elif reduction == 'sum':
        cents = th.sum(cents_per_example)
    else:
        assert reduction is None or reduction == 'none'
        cents = cents_per_example
    return cents

def soft_cross_entropy(log_preds, target, reduction='mean'):
    cents_per_example = -(target * log_preds).sum(dim=1)
    if reduction == 'mean':
        cents = th.mean(cents_per_example)
    elif reduction == 'sum':
        cents = th.sum(cents_per_example)
    else:
        assert reduction is None or reduction == 'none'
        cents = cents_per_example
    return cents