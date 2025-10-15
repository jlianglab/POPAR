

import torch.utils.data
import random
from os.path import isfile, join
from torchvision import transforms
from PIL import Image
import csv
import numpy as np
import re
import random
import copy
from torch.utils.data import Dataset
import os
from einops import rearrange
from md_aug import paint, local_pixel_shuffling,local_pixel_shuffling_500, nonlinear_transformation
import cv2
import time
from pydicom import dcmread
from os.path import isfile, join, exists
import sys


def build_md_transform(mode, dataset = "chexray"):
    transformList_mg = []
    transformList_simple = []

    if dataset == "imagenet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])


    if mode=="train":
        transformList_mg.append(local_pixel_shuffling)
        transformList_mg.append(nonlinear_transformation)
        transformList_mg.append(transforms.RandomApply([paint], p=0.9))
        transformList_mg.append(torch.from_numpy)
        transformList_mg.append(normalize)
        transformSequence_mg = transforms.Compose(transformList_mg)

        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)

        return transformSequence_mg, transformSequence_simple
    else:
        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)
        return transformSequence_simple, transformSequence_simple



class Popar_chestxray(Dataset):
    def __init__(self, image_path_file, augment, image_size=448,patch_size=32):
        self.img_list = []
        self.augment = augment
        self.patch_size = patch_size
        self.image_size = image_size
        self.graycodes = []

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline().strip()
                    if line:
                        lineItems = line.split("\t")
                        imagePath = os.path.join(pathImageDirectory, lineItems[0])
                        self.img_list.append(imagePath)


    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),(self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        gt_whole = self.augment[1](imageData)



        if random.random()<0.5:
            randperm = torch.arange(0,(self.image_size//self.patch_size)**2, dtype=torch.long)
            aug_whole = self.augment[0](imageData)
        else:
            aug_whole = gt_whole
            randperm = torch.randperm((self.image_size//self.patch_size)**2, dtype=torch.long)

        return randperm, gt_whole, aug_whole

    def __len__(self):
        return len(self.img_list)


