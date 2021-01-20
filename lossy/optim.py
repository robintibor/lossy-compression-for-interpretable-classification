import torch as th
import higher
from lossy.experiments.run import load_data, preprocess_for_glow, preprocess_for_clf


def cross_ent_loss(clf, X, y):
    clf_out = clf(X)
    cross_ent = th.nn.functional.cross_entropy(clf_out, y)
    return cross_ent


def preproc_and_to_clf(preproc, X):
    X_opt = preproc(X)
    X_opt_clf = preprocess_for_clf(X_opt)
    return X_opt_clf

def higher_loop(clf, optim_clf, data, inner_transform_fn, inner_loss_fn, copy_initial_weights):
    X_opt_clfs = []
    with higher.innerloop_ctx(clf, optim_clf, copy_initial_weights=copy_initial_weights) as (
            func_clf, func_opt):
        for X, y in data:
            X_opt_clf = inner_transform_fn(X=X)
            inner_loss = inner_loss_fn(func_clf, X_opt_clf, y)
            func_opt.step(inner_loss)
            X_opt_clfs.append(X_opt_clf)
    return func_clf, func_opt, X_opt_clfs,

def higher_loss(clf, optim_clf, data, inner_transform_fn, inner_loss_fn, outer_loss_fn,
                copy_initial_weights, copy_final_weights, only_eval_last):
    if copy_final_weights:
        assert not copy_initial_weights, "not tested this combination"


    X_opt_clfs = []
    with higher.innerloop_ctx(clf, optim_clf, copy_initial_weights=copy_initial_weights) as (
            func_clf, func_opt):
        for X, y in data:
            X_opt_clf = inner_transform_fn(X=X)
            inner_loss = inner_loss_fn(func_clf, X_opt_clf, y)
            func_opt.step(inner_loss)
            X_opt_clfs.append(X_opt_clf)

    outer_losses = []
    if only_eval_last:
        X_clf = preprocess_for_clf(X)
        X_opt_clf = X_opt_clfs[-1]
        outer_loss = outer_loss_fn(func_clf, X_opt_clf, X_clf, y)
        loss = th.stack(outer_loss)
    else:
        for (X, y), X_opt_clf in zip(data, X_opt_clfs):
            X_clf = preprocess_for_clf(X)
            outer_loss = outer_loss_fn(func_clf, X_opt_clf=X_opt_clf, X_orig_clf=X_clf,
                                       y=y)
            outer_losses.append(th.stack(outer_loss))
        loss = th.mean(th.stack(outer_losses, dim=0), dim=0)

    if copy_final_weights:
        for orig_p, func_p in zip(clf.parameters(), func_clf.parameters()):
            orig_p.data = func_p.data.detach()
            # see also https://github.com/facebookresearch/higher/issues/10#issuecomment-574695558 for alternative
        assert len(func_opt.state) == 1
        state_dict_state = {}
        for p_clf, (key, state) in zip(clf.parameters(), func_opt.state[0].items()):
            state_dict_state[p_clf] = {k: v if not hasattr(v, 'detach') else v.detach() for k, v in state.items()}
        assert len(optim_clf.param_groups) == 1
        optim_clf.param_groups[0]['params'] = list(clf.parameters())
        state_dict = dict(state=state_dict_state, param_groups=optim_clf.param_groups)
        optim_clf.load_state_dict(state_dict)
    return loss
