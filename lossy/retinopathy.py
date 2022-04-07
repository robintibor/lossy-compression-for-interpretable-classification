from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os.path
from PIL import Image
import torch
import pandas as pd

my_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



class RetinaDataset(Dataset):
    def __init__(self,
                 imagepath="/work/dlclarge2/schirrmr-lossy-compression/diabetic-retinopathy/resized/resized_train_cropped/resized_train_cropped",
                 i_start=0, i_stop=None, transform=my_transform):
        self.df = pd.read_csv(
            "/work/dlclarge2/schirrmr-lossy-compression/diabetic-retinopathy/resized/trainLabels_cropped.csv")

        self.df = self.df.iloc[i_start:i_stop]
        self.transform = transform
        self.imagepath = imagepath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.imagepath, self.df.iloc[index].image + ".jpeg")
        img = Image.open(img_path)

        if (self.transform):
            img = self.transform(img)

        return img, torch.tensor(self.df.iloc[index].level)
