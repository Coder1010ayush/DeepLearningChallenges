# ----------------------- utf-8 encoding --------------------
# this file defines function for visualizing images using matplotlib.pyplot and PIL module
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib as pth
from PIL import Image
import random


# L = R * 299/1000 + G * 587/1000 + B * 114/1000
# method to covert rgb image into grayscale according to PIL module (imp)

""" 
these are the classes present in the dataset : count is 110 
['horse racing', 'table tennis', 'cheerleading', 'figure skating men', 'sailboat racing', 'lacrosse', 
 'tug of war', 'fencing', 'hockey', 'hang gliding', 'air hockey', 'bmx', 'curling', 'hurdles', 'basketball', 
 'axe throwing', 'shot put', 'water cycling', 'bull riding', 'bobsled', 'figure skating pairs', 
 'figure skating women', 'ski jumping', 'field hockey', 'pommel horse', 'fly fishing', 'archery', 
 'steer wrestling', 'billiards', 'sumo wrestling', 'rock climbing', 'pole climbing', 'football', 
 'wheelchair basketball', 'hydroplane racing', 'rowing', 'ice yachting', 'track bicycle', 'trapeze', 
 'speed skating', 'mushing', 'horseshoe pitching', 'water polo', 'formula 1 racing', 'uneven bars', 
 'chuckwagon racing', 'harness racing', 'javelin', 'baton twirling', 'snow boarding', 'pole dancing', 
 'motorcycle racing', 'gaga', 'rings', 'horse jumping', 'judo', 'canoe slamon', 'bike polo', 'arm wrestling', 
 'wingsuit flying', 'parallel bar', 'ampute football', 'wheelchair racing', 'hammer throw', 'snowmobile racing', 
 'log rolling', 'giant slalom', 'tennis', 'roller derby', 'olympic wrestling', 'polo', 'baseball', 'cricket', 
 'bowling', 'disc golf', 'rollerblade racing', 'croquet', 'rugby', 'golf', 'bungee jumping', 'jai alai', 
 'shuffleboard', 'sky surfing', 'ice climbing', 'ultimate', 'volleyball', 'frisbee', 'luge', 'weightlifting', 
 'jousting', 'nascar racing', 'balance beam', 'swimming', 'skydiving', 'high jump', 'pole vault', 'sidecar racing', 'boxing',
 'surfing', 'barell racing']
"""


def visualizeImage(num_cnt: int = 2, num_varient: int = 5, folder: str = "/home/infinity/Documents/icpr_challenges/vggnet/sports_images/train"):
    dirs = os.listdir(path=folder)
    # print("total number of folders are ", len(dirs))
    # print("folders are ", dirs)

    # selecting random {num_varient} classes out of 110 classes
    random_dirs = random.choices(dirs, k=num_varient)
    image_path = []
    for dirPath in random_dirs:
        image_dir_path = os.path.join(folder, dirPath)
        # selecting random {num_cnt} image from the all the image of a class
        val = random.choices(os.listdir(path=image_dir_path), k=num_cnt)
        for img_path in val:
            image_path.append(image_dir_path+"/" + img_path)
    fig, axes = plt.subplots(num_varient, num_cnt, figsize=(15, num_varient * 3))
    axes = axes.flatten()

    # Plot each image
    for idx, img_path in enumerate(image_path):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(os.path.basename(img_path))
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


def showImage(path: str):
    image = Image.open(path)
    plt.imshow(image)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # showImage(path="/home/infinity/Documents/icpr_challenges/vggnet/sports_images/train/air hockey/001.jpg")
    visualizeImage()
