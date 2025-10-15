# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import random
from glob import glob
import copy
import utils
import csv


def build_classfication_dataset(dataset,data_file,transCrop, anno_percent, mode):
    if transCrop == 224:
        transResize = 256
    elif transCrop == 448:
        transResize = 512
    elif transCrop == 256:
        transResize = 293
    elif transCrop == 384:
        transResize = 440
    elif transCrop == 512:
        transResize = 585
    transform = build_classfication_transform(mode, transCrop=transCrop, transResize=transResize)

    if dataset =="ChestXray14":
        dataset =ChestX_ray14(data_file,transform, anno_percent=anno_percent)
    elif dataset =="ChexPert":
        dataset =CheXpert(data_file,transform,anno_percent=anno_percent)
    elif dataset =="ShenzhenCXR":
        dataset =ShenzhenCXR(data_file,transform,annotation_percent=anno_percent)
    elif dataset =="RSNAPneumonia":
        dataset =RSNAPneumonia(data_file,transform,annotation_percent=anno_percent)

    return dataset

def build_classfication_transform(mode="train", transCrop=224, transResize=256):

    transformList = []
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transCrop = transCrop
    transResize = transResize
    if mode == "train":
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.RandomRotation(7))
        transformList.append(transforms.ToTensor())
        if normalize is not None:
            transformList.append(normalize)
    elif mode == "validation":
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if normalize is not None:
            transformList.append(normalize)
    elif mode == "test":
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
            transformList.append(
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transformSequence = transforms.Compose(transformList)

    return transformSequence

#==============================================NIH======================================================================
class ChestX_ray14(Dataset):
    def __init__(self, image_path_file, augment, anno_percent=100):

        self.img_list = []
        self.img_label = []
        self.augment = augment

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline().strip()
                    if line:
                        lineItems = line.split()
                        imagePath = os.path.join(pathImageDirectory, lineItems[0])
                        imageLabel = lineItems[1:]
                        imageLabel = [int(i) for i in imageLabel]
                        self.img_list.append(imagePath)
                        self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')

        if self.augment != None: imageData = self.augment(imageData)

        return imageData, imageLabel

    def __len__(self):
        return len(self.img_list)


#==============================================CheXpert======================================================================
class CheXpert(Dataset):

  def __init__(self, image_path_file, augment, num_class=14,uncertain_label="LSR-Ones", unknown_label=0, anno_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label


    for pathImageDirectory, pathDatasetFile in image_path_file:
        with open(pathDatasetFile, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                imagePath = os.path.join(pathImageDirectory, line[0])
                if "test.csv" in pathDatasetFile:
                    label = line[1:]
                else:
                    label = line[5:]
                for i in range(num_class):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == 0:
                            label[i] = 0
                        elif a == -1:  # uncertain label
                            label[i] = -1
                    else:
                        label[i] = unknown_label  # unknown label

                self.img_list.append(imagePath)
                imageLabel = [int(i) for i in label]
                self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if anno_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * anno_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)




#==============================================ShenzhenCXR======================================================================

class ShenzhenCXR(Dataset):

  def __init__(self, image_path_file, augment, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    for pathImageDirectory, pathDatasetFile in image_path_file:
        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline().strip()
                if line:
                    lineItems = line.split(',')
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

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

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


#==============================================RSNAPneumonia======================================================================

class RSNAPneumonia(Dataset):

  def __init__(self, image_path_file, augment, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    for pathImageDirectory, pathDatasetFile in image_path_file:
        with open(pathDatasetFile, "r") as fileDescriptor:
          line = True

          while line:
            line = fileDescriptor.readline()
            if line:
              lineItems = line.strip().split(' ')
              imagePath = os.path.join(pathImageDirectory, lineItems[0])
              self.img_list.append(imagePath)
              self.img_label.append(int(lineItems[-1]))

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

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = self.img_label[index]
    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)







class MLP_Xpert_Single_Gender(Dataset):
    def __init__(self, input_embedding_path, csv_path):
        self.input_embedding = np.load(input_embedding_path,allow_pickle=True).item()
        self.img_label = []
        self.image_path = []
        with open(csv_path) as fr:
            csv_file = csv.reader(fr)
            next(csv_file)
            for line in csv_file:
                self.image_path.append( line[1].replace("-small","").replace("/","-").replace(".jpg",""))
                label = line[4:]
                imageLabel = [float(i) for i in label]
                self.img_label.append(imageLabel)



    def __getitem__(self, index):
        imageData = self.input_embedding[self.image_path[index]]
        imageLabel = torch.FloatTensor(self.img_label[index])

        return imageData, imageLabel
    def __len__(self):
        return len(self.image_path)



class MLP_Nih14_Single_Gender(Dataset):
    def __init__(self, input_embedding_path, csv_path):

        self.input_embedding = np.load(input_embedding_path,allow_pickle=True).item()
        self.img_label = []
        self.image_path = []

        with open(csv_path) as fr:
            csv_file = csv.reader(fr)
            next(csv_file)
            for line in csv_file:
                self.image_path.append(line[1].replace(".png",""))
                imageLabel = line[12:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_label.append(imageLabel)


    def __getitem__(self, index):
        imageData = self.input_embedding[self.image_path[index]]
        imageLabel = torch.FloatTensor(self.img_label[index])


        return imageData, imageLabel
    def __len__(self):
        return len(self.image_path)
