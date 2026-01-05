import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from glob import glob
from tqdm import tqdm
from skimage.transform import resize
import os
import random

def minmax_scaler_np(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def minmax_scaler(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

class MyDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        imgs = np.load(data_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")

        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = np.array(img)

        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)

class FFTDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        imgs = np.load(data_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")

        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = np.array(img)
        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        if self.transform is not None:
            int_img = self.transform(img)

        return img, int(label), int_img


class INTDataset(Dataset):
    def __init__(self, data_path, data_path_int, label_path, transform=None):
        imgs = np.load(data_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        imgs_int = np.load(data_path_int, mmap_mode="r")

        self.imgs = imgs
        self.imgs_int = imgs_int
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        img_int = self.imgs_int[index]
        label = self.labels[index]

        img = np.array(img)
        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        img_int = np.array(img_int)
        img_int = img_int.astype('float32')
        img_int = np.expand_dims(img_int, axis=0)

        if self.transform is not None:
            int_img = self.transform(img)

        return img, int(label), img_int