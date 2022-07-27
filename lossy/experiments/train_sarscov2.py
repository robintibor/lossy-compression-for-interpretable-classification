import logging
import os.path

import numpy as np
import pandas as pd
import torch as th
from rtsutils.nb_util import TerminalResults
from rtsutils.util import th_to_np
from torch import nn
from torchvision import transforms
from tqdm import tqdm, trange

from lossy.sars_cov_2 import SarsCov2Scans
from lossy.util import set_random_seeds
import sklearn.metrics

log = logging.getLogger(__name__)


def run_exp(n_epochs, np_th_seed, output_dir,):
    set_random_seeds(np_th_seed, True)
    log.info("Load data...")
    unzip_df = pd.read_csv('/work/dlclarge2/schirrmr-lossy-compression/sars-cov-2-ct/unzip_filenames.csv')
    unzip_df = unzip_df[unzip_df.zip_file != 'NCP-13.zip']  # exclude broken zip file
    rng = np.random.RandomState(20220704)
    shuffled_inds = rng.permutation(len(unzip_df))
    n_split = int(0.75 * len(shuffled_inds))
    i_train = shuffled_inds[:n_split]
    i_test = shuffled_inds[n_split:]

    train_df = unzip_df.iloc[i_train]
    test_df = unzip_df.iloc[i_test]

    from lossy.augment import Choice

    train_transforms = transforms.Compose((
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=8, padding_mode='reflect'),
        Choice(
            (
                # (transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(20, shear=20),
            ), ),
        transforms.Lambda(lambda im: im.convert('RGB'), ),
        transforms.ToTensor()
    ))
    test_transforms = transforms.Compose((
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda im: im.convert('RGB'), ),
        transforms.ToTensor()
    ))

    train_set = SarsCov2Scans('/work/dlclarge2/schirrmr-lossy-compression/sars-cov-2-ct/', train_df, 1, 'random',
                              transform=train_transforms)
    test_set = SarsCov2Scans('/work/dlclarge2/schirrmr-lossy-compression/sars-cov-2-ct/', test_df, 1, 'fixed',
                             transform=test_transforms)

    train_loader = th.utils.data.DataLoader(
        train_set,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        batch_size=64,
    )
    test_loader = th.utils.data.DataLoader(
        test_set,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        batch_size=64,
    )


    log.info("Load model...")
    normalize_rad_imagenet = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[1, 1, 1])
    model = th.load('/work/dlclarge2/schirrmr-lossy-compression/classifiers/radimagenet/RadImageNet-ResNet50-notop.pt')
    feature_model = nn.Sequential(normalize_rad_imagenet, model).cuda().eval()

    clf_head = nn.Linear(2048, 3, device='cuda')
    clf = nn.Sequential(feature_model, clf_head)
    opt_clf = th.optim.AdamW(clf.parameters(), lr=3e-4, weight_decay=1e-4)

    log.info("Run training...")
    import sklearn
    tqdm_obj = trange(n_epochs)
    nb_res = TerminalResults(tqdm_obj, 0.95)
    best_acc = -1
    for i_epoch in trange(n_epochs):
        clf.train()
        for X, y in train_loader:
            X = X.cuda()
            y = y.cuda()
            out = clf(X)
            loss = th.nn.functional.cross_entropy(out, y)
            opt_clf.zero_grad(set_to_none=True)
            loss.backward()
            opt_clf.step()
            opt_clf.zero_grad(set_to_none=True)
            nb_res.collect(
                loss=loss.item())
            nb_res.print()
        clf.eval()
        if (i_epoch % max(n_epochs // 20, 1) == 0) or (i_epoch == (n_epochs - 1)):
            pred_df = pd.DataFrame()
            for X, y in tqdm(test_loader):
                X = X.cuda()
                with th.no_grad():
                    out = clf(X)
                pred_df = pd.concat((
                    pred_df,
                    pd.DataFrame(dict(y=list(th_to_np(y)),
                                      out=list(th_to_np(out))))))

            test_y = np.array(pred_df.y)
            test_preds = np.stack(pred_df.out)
            acc = (test_preds.argmax(axis=1) == test_y).mean()

            balanced_acc = sklearn.metrics.balanced_accuracy_score(test_y, test_preds.argmax(axis=1))

            auc = np.mean([sklearn.metrics.roc_auc_score(test_y == i_class,
                                                         test_preds[:, i_class]) for i_class in range(3)])
            print(f"Epoch {i_epoch}")
            print(f"Acc:          {acc:.1%}")
            print(f"Balanced Acc: {balanced_acc:.1%}")
            print(f"AUC:          {auc:.1%}")
            th.save(clf, os.path.join(output_dir, f"clf_{i_epoch:d}.th"))
            if acc > best_acc:
                best_acc = acc
                th.save(clf, os.path.join(output_dir, "best_clf.th"))

    #th.save(clf, os.path.join(output_dir, "clf.th"))

    results = dict(
        loss=loss.item(),
        acc=acc.item(),
        balanced_acc=balanced_acc.item(),
        auc=auc.item(),
    )
    return results
