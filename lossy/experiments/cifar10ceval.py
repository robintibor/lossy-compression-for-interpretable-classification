import json
import os.path

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm
from glob import glob

from lossy.classifier import get_classifier_from_folder
from lossy.preproc import get_preprocessor_from_folder
from lossy.util import np_to_th, th_to_np


def run_exp(saved_exp_folder, with_preproc):
    sets = [os.path.split(s)[-1] for s in glob("/home/schirrmr/data/CIFAR-10-C/*.npy")]
    sets = [s for s in sets if s != "labels.npy"]
    set_folder = "/home/schirrmr/data/CIFAR-10-C/"
    corrupted_y = np.load(
        "/home/schirrmr/data/CIFAR-10-C/labels.npy",
    )
    corrupted_out_file_name = os.path.join(
        saved_exp_folder, f"corrupted_dfs_logits_with{['out', ''][with_preproc]}_preproc.pkl.npy"
    )
    if not os.path.exists(corrupted_out_file_name):
        config = json.load(open(os.path.join(saved_exp_folder, "config.json"), "r"))
        clf_exp_folder = saved_exp_folder
        if "saved_exp_folder" in config:
            original_exp_folder = config["saved_exp_folder"]
        else:
            original_exp_folder = saved_exp_folder

        clf = get_classifier_from_folder(original_exp_folder).eval()
        clf.load_state_dict(th.load(os.path.join(clf_exp_folder, "clf_state_dict.th")))
        preproc = get_preprocessor_from_folder(original_exp_folder).eval()

        setname_to_df = {}
        for setname in tqdm(sets):
            print(" ".join([s.capitalize() for s in setname.split(".")[0].split("_")]))
            X_npy = np.load(os.path.join(set_folder, setname))
            corrupted_set = th.utils.data.TensorDataset(
                np_to_th(X_npy.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
            )
            corrupted_loader = th.utils.data.DataLoader(
                corrupted_set,
                batch_size=128,
                shuffle=False,
                drop_last=False,
            )
            set_df = pd.DataFrame()
            for (X,) in tqdm(corrupted_loader):
                with th.no_grad():
                    X = X.cuda()
                    if with_preproc:
                        X = preproc(X)
                    out = clf(X)
                set_df = pd.concat(
                    (
                        set_df,
                        pd.DataFrame(
                            dict(
                                clf_out_labels=th_to_np(out).argmax(axis=1),
                                clf_outs=list(th_to_np(out).astype(np.float32)),
                            )
                        ),
                    )
                )

            setname_to_df[setname] = set_df
        np.save(corrupted_out_file_name, setname_to_df)
    setname_to_df = np.load(corrupted_out_file_name, allow_pickle=True).item()
    means_per_set = {
        setname: np.mean(df.clf_out_labels == corrupted_y)
        for setname, df in setname_to_df.items()
    }
    results = {**means_per_set, **{"avg": np.mean(list(means_per_set.values()))}}
    return results
