# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from os.path import isfile, join
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import random
import cv2
from einops import rearrange
from torchvision import transforms
import os
import json
import csv
import copy
import albumentations
from albumentations import Compose, HorizontalFlip, Normalize, VerticalFlip, Rotate, Resize, ShiftScaleRotate, OneOf, GridDistortion, OpticalDistortion, \
    ElasticTransform,RandomBrightnessContrast,RandomGamma,RandomSizedCrop,RandomContrast,RandomBrightness

from albumentations.pytorch import ToTensorV2




class MontgomeryDataset(Dataset):

    def __init__(self, image_path_file, image_size=(448,448), mode= "train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/CXR_png", line))
                    self.img_label.append(
                        (join(pathImageDirectory+"/ManualMask/leftMask", line),(join(pathImageDirectory+"/ManualMask/rightMask", line)))
                         )
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData + rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask


class JSRTLungDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), mode="train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode


        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/images", line+".IMG.png"))
                    self.img_label.append(
                        (join(pathImageDirectory+"/masks/left_lung_png", line+".png"),(join(pathImageDirectory+"/masks/right_lung_png", line+".png")))
                         )
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]



        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255



        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData + rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask


class JSRTClavicleDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), mode="train", annotation_percent = 100):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }



        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/images", line+".IMG.png"))
                    self.img_label.append(
                        (join(pathImageDirectory+"/masks/left_clavicle_png/", line+".png"),(join(pathImageDirectory+"/masks/right_clavicle_png/", line+".png")))
                         )
                    line = fileDescriptor.readline().strip()

        indexes = np.arange(len(self.img_list))
        if annotation_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData + rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask

class JSRTHeartDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), mode="train"):
        self.img_list = []
        self.img_label = []
        self.image_size = image_size

        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }



        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/images", line+".IMG.png"))
                    self.img_label.append(join(pathImageDirectory+"/masks/heart_png/", line+".png"))
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        maskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData[maskData>0] =255
        maskData = maskData/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask

class VinDrRibCXRDataset(Dataset):
    def __init__(self, image_path_file, image_size, mode, anno_percent=100):
        self.pathImageDirectory, pathDatasetFile = image_path_file
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        self.rib_labels =  ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10',
                           'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
        f = open(pathDatasetFile)
        data= json.load(f)

        self.img_list = data['img']
        self.label_list = data

        self.indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(self.indexes)
            num_data = int(self.indexes.shape[0] * anno_percent / 100.0)
            self.indexes = self.indexes[:num_data]


    def __getitem__(self, index):
        index = self.indexes[index]
        imagePath = self.img_list[str(index)]
        imageData = cv2.imread(os.path.join(self.pathImageDirectory, imagePath), cv2.IMREAD_COLOR)
        label0 = []
        for name in self.rib_labels:
            pts = self.label_list[name][str(index)]
            label = np.zeros((imageData.shape[:2]), dtype=np.uint8)
            if pts != 'None':
                pts = np.array([[[int(pt['x']), int(pt['y'])]] for pt in pts])
                label = cv2.fillPoly(label, [pts], 1)
                label = cv2.resize(label, self.image_size,interpolation=cv2.INTER_AREA)
            label0.append(label)
        label0 = np.stack(label0)
        label0 = label0.transpose((1, 2, 0))

        imageData = cv2.resize(imageData,self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode] (image=imageData, mask=label0)
        img = dic['image']
        mask = (dic['mask'].permute(2, 0, 1))

        return img, mask



    def __len__(self):
        return len(self.indexes)



class LinearProbeDataset(Dataset):
    def __init__(self, input_embedding_path, input_lbl_path, dataset):
        self.input_embedding = np.load(input_embedding_path)
        self.input_lbl = np.load(input_lbl_path)
        self.random_shuffle()
        self.dataset = dataset

    def __getitem__(self, index):
        imageData = self.input_embedding[index]


        if self.dataset == "chexpert":
            label = []
            for l in self.input_lbl[index]:
                if l == -1:
                    label.append(random.uniform(0.55, 0.85))
                else:
                    label.append(l)
            imageLabel = torch.FloatTensor(label)
        else:
            imageLabel = torch.FloatTensor(self.input_lbl[index])

        return imageData, imageLabel

    def random_shuffle(self):
        idx = np.random.permutation(range(len(self.input_embedding)))
        self.input_embedding = self.input_embedding[idx,:]
        self.input_lbl = self.input_lbl[idx,:]

    def __len__(self):
        return len(self.input_embedding)