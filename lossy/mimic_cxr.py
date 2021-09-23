from PIL import Image
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
import pandas as pd
import os.path
import numpy as np
from torchvision import transforms


class MIMIC_CXR_JPG(VisionDataset):
    def __init__(
        self,
        root: str,
        target: str,
        split: str,
        n_dicoms: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        mimic_folder = root

        if target == "race":
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
            wanted_subject_df.loc[:, "label"] = wanted_subject_df.ethnicity.replace(
                {
                    "ASIAN": 0,
                    "BLACK/AFRICAN AMERICAN": 1,
                    "WHITE": 2,
                }
            )
            label_df = wanted_subject_df.copy()
            merge_label_on = "subject_id"
            class_names = ["ASIAN", "BLACK/AFRICAN AMERICAN", "WHITE"]
        elif target == "disease":
            chexpert_df = pd.read_csv(
                os.path.join(mimic_folder, "mimic-cxr-2.0.0-chexpert.csv")
            )
            mask_nothing = (
                (chexpert_df.loc[:, "Cardiomegaly"] == 0)
                & (chexpert_df.loc[:, "Pleural Effusion"] == 0)
                & (chexpert_df.loc[:, "No Finding"] == 1)
            )
            mask_cardiomegaly = (chexpert_df.loc[:, "Cardiomegaly"] == 1) & (
                chexpert_df.loc[:, "Pleural Effusion"] == 0
            )
            mask_pleural = (chexpert_df.loc[:, "Cardiomegaly"] == 0) & (
                chexpert_df.loc[:, "Pleural Effusion"] == 1
            )
            mask_select = mask_nothing | mask_cardiomegaly | mask_pleural
            label_arr = np.full(len(mask_nothing), -1)
            label_arr = label_arr + np.sum(
                [
                    mask * (i + 1)
                    for i, mask in enumerate(
                        [mask_nothing, mask_cardiomegaly, mask_pleural]
                    )
                ],
                axis=0,
            )

            assert np.all((label_arr != -1) == mask_select)
            chexpert_df.loc[:, "label"] = label_arr
            label_df = chexpert_df.copy()
            merge_label_on = ["study_id", "subject_id"]
            class_names = ["No Finding", "Cardiomegaly", "Pleural Effusion"]
        elif target == "pleural_effusion":
            chexpert_df = pd.read_csv(
                os.path.join(mimic_folder, "mimic-cxr-2.0.0-chexpert.csv")
            )
            mask_nothing = (chexpert_df.loc[:, "Pleural Effusion"] == 0) & (
                chexpert_df.loc[:, "No Finding"] == 1
            )

            mask_pleural = chexpert_df.loc[:, "Pleural Effusion"] == 1

            mask_select = mask_nothing | mask_pleural
            label_arr = np.full(len(mask_nothing), -1)
            label_arr = label_arr + np.sum(
                [mask * (i + 1) for i, mask in enumerate([mask_nothing, mask_pleural])],
                axis=0,
            )

            assert np.all((label_arr != -1) == mask_select)
            chexpert_df.loc[:, "label"] = label_arr
            label_df = chexpert_df.copy()
            merge_label_on = ["study_id", "subject_id"]
            class_names = ["No Finding", "Pleural Effusion"]
        elif target == "cardiomegaly":
            chexpert_df = pd.read_csv(
                os.path.join(mimic_folder, "mimic-cxr-2.0.0-chexpert.csv")
            )
            mask_nothing = (chexpert_df.loc[:, "Pleural Effusion"] == 0) & (
                chexpert_df.loc[:, "No Finding"] == 1
            )

            mask_cardiomegaly = chexpert_df.loc[:, "Cardiomegaly"] == 1

            mask_select = mask_nothing | mask_cardiomegaly
            label_arr = np.full(len(mask_nothing), -1)
            label_arr = label_arr + np.sum(
                [
                    mask * (i + 1)
                    for i, mask in enumerate([mask_nothing, mask_cardiomegaly])
                ],
                axis=0,
            )

            assert np.all((label_arr != -1) == mask_select)
            chexpert_df.loc[:, "label"] = label_arr
            label_df = chexpert_df.copy()
            merge_label_on = ["study_id", "subject_id"]
            class_names = ["No Finding", "Cardiomegaly"]
        elif target == "age":
            patients_df = pd.read_csv(os.path.join(mimic_folder, "patients.csv"))
            mask_20_40 = (patients_df.anchor_age >= 20) & (patients_df.anchor_age < 40)
            mask_40_60 = (patients_df.anchor_age >= 40) & (patients_df.anchor_age < 60)
            mask_60_85 = (patients_df.anchor_age >= 60) & (patients_df.anchor_age < 85)
            mask_select = mask_20_40 | mask_40_60 | mask_60_85
            label_arr = np.full(len(mask_select), -1)
            label_arr = label_arr + np.sum(
                [
                    mask * (i + 1)
                    for i, mask in enumerate([mask_20_40, mask_40_60, mask_60_85])
                ],
                axis=0,
            )
            assert np.all((label_arr != -1) == (mask_select))
            patients_df.loc[:, "label"] = label_arr
            label_df = patients_df.copy()
            merge_label_on = "subject_id"
            class_names = t
        elif target == "gender":
            patients_df = pd.read_csv(os.path.join(mimic_folder, "patients.csv"))
            label_arr = patients_df.gender.replace(dict(F=0, M=1))
            patients_df.loc[:, "label"] = label_arr
            assert np.array_equal(np.unique(label_arr), [0, 1])
            label_df = patients_df.copy()
            merge_label_on = "subject_id"
            class_names = [
                "Female",
                "Male",
            ]
        else:
            assert False

        dicom_df = pd.read_csv(os.path.join(mimic_folder, "mimic-cxr-2.0.0-split.csv"))
        full_df = dicom_df.merge(label_df, on=merge_label_on)
        full_df = full_df[full_df.label != -1]

        # Make set balanced between classes,
        # so determine number of examples by class with least examples
        labels = np.unique(full_df.label)
        class_dfs = [full_df[full_df.label == i] for i in labels]
        n_per_class = min(min(len(d) for d in class_dfs), n_dicoms // len(labels))
        full_df = (
            pd.concat([d.iloc[:n_per_class].reset_index(drop=True) for d in class_dfs])
            .sort_index()
            .reset_index(drop=True)
        )

        # full_df = full_df.iloc[:n_dicoms]
        # let's create new split for now
        subject_ids = sorted(full_df.subject_id.unique())
        new_split = np.concatenate(
            [["train"] * 5 + ["validate"]] * ((len(subject_ids) + 6) // 6)
        )[: len(subject_ids)]
        new_split_df = pd.DataFrame(dict(subject_id=subject_ids, split=new_split))
        new_full_df = full_df.drop("split", axis=1).merge(new_split_df, on="subject_id")
        this_split_df = new_full_df[new_full_df.split == split].sort_values(
            by="subject_id"
        )

        self.set_df = this_split_df.copy()
        self.classes = class_names

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
            img_32_32 = transforms.Resize(
                (32, 32), transforms.InterpolationMode.BILINEAR
            )(img)
            img_32_32.save(jpg_32_file)
        img = Image.open(jpg_32_file)
        target = row["label"]
        assert target >= 0

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.set_df)
