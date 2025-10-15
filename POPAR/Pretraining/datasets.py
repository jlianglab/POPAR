

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

ALL_XRAYS={
    'nih14':["DATA_FOLDER/nih_xray14/images/images","DATA_FOLDER/nih_xray14/images/images", '.png'],
    'jsrt':["DATA_FOLDER/JSRT/All247images/images/","DATA_FOLDER/JSRT/All247images/images/",".png"],
    'mendeleyv2':["DATA_FOLDER/Mendeley-V2/CellData/chest_xray/","DATA_FOLDER/Mendeley-V2/CellData/chest_xray/",".jpeg"],
    'montgomery': ["DATA_FOLDER/MontgomeryCountyX-ray/MontgomerySet/CXR_png/", "DATA_FOLDER/MontgomerySet/CXR_png/", ".png"],
    'shenzhen': ["DATA_FOLDER/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/", "/data/jliang12/mhossei2/Dataset/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/", ".png"],
    'rsna': ["DATA_FOLDER/rsna-pneumonia-detection-challenge/", "DATA_FOLDER/rsna-pneumonia-detection-challenge/", ".png"],
    'chexpert': ["DATA_FOLDER/CheXpert-v1.0/", "/data/jliang12/mhossei2/Dataset/CheXpert-v1.0/", ".jpg"],
    'padchest': ["DATA_FOLDER/PadChest/image_zips", "DATA_FOLDER/PadChest/image_zips/", ".png"],
    'mimiccxr': ["DATA_FOLDER/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0", "DATA_FOLDER/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0", ".jpg"],
    'indiana': ["DATA_FOLDER/Indiana_ChestX-ray/images/images_normalized/", "DATA_FOLDER/Indiana_ChestX-ray/images/images_normalized/", ".jpeg"],
    'convidx': ["DATA_FOLDER/COVIDx/", "DATA_FOLDER/COVIDx/", ".png"],
    'convidradiography': ["DATA_FOLDER/COVID-19_Radiography_Dataset/", "DATA_FOLDER/COVID-19_Radiography_Dataset/", ".png"],
    'vindrcxr': ["DATA_FOLDER/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/", "DATA_FOLDER/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/", ".jpeg"],
    'rocird_covid': ["DATA_FOLDER/rocird_covid/png", "DATA_FOLDER/RICORD_covid/png", ".png"],
    'nih_tb_portals': ["DATA_FOLDER/nih_tb_portal", "DATA_FOLDER/nih_tb_portals",  ".png"],
    'plco': ["DATA_FOLDER/PLCOI-880", "DATA_FOLDER/PLCOI-880", ".jpeg"],

}



class MaskGenerator:
    def __init__(self, input_size=448, mask_patch_size=16, model_patch_size=4, mask_ratio=0.5):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def get_mask(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask



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




def build_simple_transform(dataset = "chexray"):
    transformList = []

    if dataset == "imagenet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])

    transformList.append(torch.from_numpy)
    transformList.append(normalize)
    transformSequence = transforms.Compose(transformList)


    return transformSequence




class Popar_allXray(Dataset):
    def __init__(self,include_test_data=False, augment = None, machine="lab", use_rgb = True, channel_first = True, writter = sys.stdout, views = "frontal", image_size=448,patch_size=32):
        self.img_list = []
        self.augment = augment
        self.use_rgb =use_rgb
        self.channel_first = channel_first
        self.image_size = image_size
        self.patch_size = patch_size
        if views=="frontal":
            self.data_index = join("data_index", "frontal")
        else:
            self.data_index = join("data_index", "all")

        if machine =="lab":
            _indictor = 0
        else:
            _indictor = 1
        self.writter = writter
        for i, (key, value) in enumerate(ALL_XRAYS.items()):
            print("Initializing [{}/{}]: {} dataset".format(i+1,len(ALL_XRAYS),key),file = self.writter)
            with open(join(join(self.data_index,key), "train.txt"), 'r') as fr:
                line = fr.readline()
                while line:
                    self.img_list.append([join(value[_indictor], line.split(' ')[0].strip()), value[-1]])
                    line = fr.readline()
            if include_test_data and exists(join(key, "test.txt")):
                with open(join(join(self.data_index,key), "test.txt"), 'r') as fr:
                    line = fr.readline()
                    while line:
                        self.img_list.append([join(value[_indictor], line.split(' ')[0].strip()), value[-1]])
                        line = fr.readline()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path_info = self.img_list[index]
        try:
            if "dcm" in img_path_info[-1]:
                img_data = dcmread(img_path_info[0]).pixel_array
            else:
                img_data = cv2.imread(img_path_info[0], cv2.IMREAD_GRAYSCALE)


            if img_data is None:
                self.__getitem__(random.randrange(0, index))
            img_data = cv2.resize(img_data,(self.image_size, self.image_size),interpolation=cv2.INTER_AREA)

            if np.min(img_data)<0 or np.max(img_data)>255:
                img_data = cv2.normalize(src=img_data, dst=img_data, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            img_data = img_data.astype(np.uint8)
            if self.use_rgb:
                img_data = np.repeat(img_data[:, :, np.newaxis], 3, axis=2)
                if self.channel_first:
                    img_data = rearrange(img_data, 'h w c ->c h w')

            img_data = img_data / 255

            gt_whole = self.augment[1](img_data)
            if random.random() < 0.5:
                randperm = torch.arange(0, (self.image_size // self.patch_size) ** 2, dtype=torch.long)
                aug_whole = self.augment[0](img_data)
            else:
                aug_whole = gt_whole
                randperm = torch.randperm((self.image_size // self.patch_size) ** 2, dtype=torch.long)
            return randperm, gt_whole, aug_whole

        except Exception as e:
            print("error in: ", img_path_info[0])
            print(e,file = self.writter)
            self.__getitem__( random.randrange(0, index))




class Popar_chestxray(Dataset):
    def __init__(self, image_path_file, augment, image_size=448,patch_size=32):
        self.img_list = []
        self.augment = augment
        self.patch_size = patch_size
        self.image_size = image_size

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline().strip()
                    if line:
                        lineItems = line.split(" ")
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





class Popar_imagenet(Dataset):
    def __init__(self, image_path_file, augment, image_size=448,patch_size=32):
        self.img_list = []
        self.augment = augment
        self.patch_size = patch_size
        self.image_size = image_size

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline()
                    if line:
                        lineItems = line.split()
                        imagePath = os.path.join(pathImageDirectory, lineItems[2])
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







