from PIL import Image
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
import pandas as pd
import os.path
import numpy as np
from torchvision  import transforms


class MIMIC_CXR_JPG(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        n_dicoms: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        mimic_folder = root

        admissions_df = pd.read_csv(os.path.join(mimic_folder, "admissions.csv"))
        ethnicity_df = admissions_df.loc[
            :, ["subject_id", "ethnicity"]
        ].drop_duplicates()

        v = ethnicity_df.subject_id.value_counts()
        subject_id_more_than_once = v.index[v.gt(1)]
        unambigous_ethnicity_df = ethnicity_df[
            ~ethnicity_df.subject_id.isin(subject_id_more_than_once)
        ]
        assert unambigous_ethnicity_df.subject_id.value_counts().gt(1).sum() == 0

        wanted_ethnicity_df = unambigous_ethnicity_df[
            unambigous_ethnicity_df.ethnicity.isin(
                ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN"]
            )
        ]

        patients_df = pd.read_csv(os.path.join(mimic_folder, "patients.csv"))
        gender_df = patients_df.loc[:, ["subject_id", "gender"]]
        wanted_subject_df = wanted_ethnicity_df.merge(
            gender_df[gender_df.subject_id.isin(wanted_ethnicity_df.subject_id)],
            on="subject_id",
        ).sort_values(by="subject_id")
        dicom_df = pd.read_csv(os.path.join(mimic_folder, "mimic-cxr-2.0.0-split.csv"))
        full_df = dicom_df[
            dicom_df.subject_id.isin(wanted_subject_df.subject_id)
        ].merge(wanted_subject_df, on="subject_id").sort_values(by="subject_id")

        # let's create new split from within train for now
        subject_ids = sorted(full_df.subject_id.unique())
        new_split = np.concatenate([['train'] * 5 + ['validate']] * ((len(subject_ids) + 6) // 6))[:len(subject_ids)]
        new_split_df = pd.DataFrame(dict(subject_id=subject_ids, split=new_split))
        new_full_df = full_df.drop('split', axis=1).merge(new_split_df, on='subject_id')
        this_split_df = new_full_df[new_full_df.split == split].sort_values(by="subject_id")

        # make a balanced set and interleave asian/black/white,
        # so that later calls to subset will still yield balanced subsets
        # also for now make a trick do not use validate split,
        # rather use later part in train
        n_asian_dicoms = n_dicoms // 3
        n_black_dicoms = n_asian_dicoms
        n_white_dicoms = n_dicoms - n_asian_dicoms - n_black_dicoms

        asian_df = this_split_df[this_split_df.ethnicity == "ASIAN"].iloc[:n_asian_dicoms]
        black_df = this_split_df[this_split_df.ethnicity == "BLACK/AFRICAN AMERICAN"].iloc[
            :n_black_dicoms
        ]
        white_df = this_split_df[this_split_df.ethnicity == "WHITE"].iloc[:n_white_dicoms]

        set_df = (
            pd.concat(
                (
                    asian_df.reset_index(drop=True),
                    black_df.reset_index(drop=True),
                    white_df.reset_index(drop=True),
                )
            )
            .sort_index()
            .reset_index(drop=True)
        )

        self.set_df = set_df
        self.classes = ["ASIAN", "BLACK/AFRICAN AMERICAN", "WHITE"]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        row = self.set_df.iloc[index]
        study_folder = f"p{str(row['subject_id'])[:2]}/p{str(row['subject_id'])}/s{str(row['study_id'])}"
        study_path = os.path.join(self.root, "files", study_folder)
        jpg_dicom_file = os.path.join(study_path, f"{row['dicom_id']}.jpg")
        # Create a jpg 32x32 file for faster loading, if not exists yet
        jpg_32_file = os.path.join(study_path, f"{row['dicom_id']}.32x32.jpg")
        if not os.path.exists(jpg_32_file):
            img = Image.open(jpg_dicom_file)
            img_32_32 = transforms.Resize((32, 32), transforms.InterpolationMode.BILINEAR)(img)
            img_32_32.save(jpg_32_file)
        img = Image.open(jpg_32_file)
        target = self.classes.index(row["ethnicity"])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.set_df)
