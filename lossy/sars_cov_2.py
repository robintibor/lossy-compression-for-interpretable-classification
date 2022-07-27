import os.path
from glob import glob

import numpy as np
import torch as th
from PIL import Image
from torchvision.datasets import VisionDataset


class SarsCov2Scans(VisionDataset):
    def __init__(
        self,
        root,
        df,
        n_slices,
        fixed_or_random_slice,
        transform=None,
        target_transform=None,
    ):
        self.df = df
        self.n_slices = n_slices
        self.fixed_or_random_slice = fixed_or_random_slice
        super().__init__(
            root,
            transforms=None,
            transform=transform,
            target_transform=target_transform,
        )
        self.classes = ["NCP", "CP", "Normal"]

    def __getitem__(self, index):
        i_row = index // self.n_slices
        i_from_slice = index % self.n_slices
        row = self.df.iloc[i_row]
        if self.fixed_or_random_slice == "fixed":
            if self.n_slices == 1:
                i_slice = row.n_slice // 2
            else:
                i_slice = np.round(np.linspace(row.n_slice // 3, (row.n_slice * 2) // 3, self.n_slices)).astype(
                    int
                )[i_from_slice]
        else:
            assert self.fixed_or_random_slice == "random"
            i_slice = th.randint(0, row.n_slice, (1,)).item()
        base_folder = os.path.join(
            self.root,
            row.label,
            str(row.patient_id),
            str(row.scan_id),
        )
        image_path = glob(os.path.join(base_folder, "*"))[i_slice]
        im = Image.open(image_path)
        y = dict(NCP=0, CP=1, Normal=2)[row.label]

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return im, y

    def __len__(self):
        return len(self.df) * self.n_slices
