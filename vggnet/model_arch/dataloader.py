# ------------------------------------ utf-8 encoding ------------------------------
import torch
import torchvision
import os
import sys
import pathlib
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from preprocessing import resizeImage


class VNetDataClass(Dataset):

    def __init__(self, folder_path) -> None:
        super().__init__()
        folder_list = os.listdir(path=folder_path)
        encoder = LabelEncoder()
        labels = encoder.fit_transform(folder_list)
        label_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
        # print("label mapping is ", label_mapping)
        self.json_data = []
        for classes in folder_list:
            cache_folder_path = folder_path + "/"+classes
            images = os.listdir(cache_folder_path)
            for img in images:
                if img.endswith(".lnk"):
                    continue

                diction = {
                    "path": cache_folder_path+"/"+img,
                    "label": label_mapping[classes]
                }
                self.json_data.append(diction)
        # print(self.json_data)

    def __len__(self):
        assert len(self.json_data) > 0, "length must be greater than 0"
        return len(self.json_data)

    def __getitem__(self, index):
        dicton = self.json_data[index]
        try:
            image = resizeImage(dicton["path"], (224, 224))
            if image.shape[0] < 3:
                image = image.expand(3, 224, 224)

            label = dicton["label"]
            return image, label
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    vbc = VNetDataClass(folder_path="vggnet/sports_images/train")
    train_loader = DataLoader(vbc, 32, True)
