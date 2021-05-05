import os
from os.path import join
import numpy as np
import random
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset

import dicts

class PlantCLEF(Dataset):
    def __init__(self,path, train = True, transform = None):
        self.transform = transform
        self.dataset = datasets.ImageFolder(os.path.join(path, 'train' if train else 'val'), self.transform)
        self.idx_to_class = {v: k for k, v in self.dataset.class_to_idx.items()}

    def __getitem__(self,index):
        item = self.dataset[index]
        img = item[0]
        class_idx = item[1]
        _class = self.idx_to_class[class_idx]
        family = dicts.family[_class]
        genus = dicts.genus[_class]
        return img, family, genus, class_idx

    def __len__(self):
        return len(self.dataset)