# ------------------------------ utf-8 encoding
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from PIL import Image
from torchvision import transforms
import torchvision
import PIL as pil
import seaborn as sns
import pathlib
import sys
import json
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# this file contains function for resizing images and convert image into tensor


def decodeImage(path):
    img = Image.open(path)
    transform = transforms.Compose(
        [transforms.Resize((60, 60)),
         transforms.ToTensor()
         ]
    )

    image = transform(img)
    return image


def resizeImage(path: str, shape: tuple):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    if not path.lower().endswith(valid_extensions):
        raise UnidentifiedImageError(f"File {path} is not a recognized image format")

    try:
        img = Image.open(path)
        transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor()
        ])

        image = transform(img)
        return image
    except UnidentifiedImageError:
        raise UnidentifiedImageError(f"Cannot identify image file {path}")
    except Exception as e:
        print(f"Error occurred while processing file {path}: {e}")
        return None


if __name__ == "__main__":
    image = decodeImage(path="vggnet/sports_images/train/air hockey/001.jpg")
    image = resizeImage(
        "vggnet/sports_images/train/air hockey/001.jpg", (120, 120))
    print("imaeg tensor is ", image)
