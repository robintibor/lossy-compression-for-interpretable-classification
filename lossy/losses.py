import torch as th

def kl_divergence(clf_out_a, clf_out_b):
    return th.mean(
            th.sum(
                th.nn.functional.softmax(clf_out_a, dim=1) *
                (th.nn.functional.log_softmax(clf_out_a, dim=1) -
                 th.nn.functional.log_softmax(clf_out_b, dim=1)), dim=1))


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