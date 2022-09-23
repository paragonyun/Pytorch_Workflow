from torchvision.datasets import MNIST
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import load_config

import torch

from torch.utils.data import Dataset


import matplotlib.pyplot as plt


class CustomDataset(Dataset) :
    def __init__(self, img, labels, config) :
        self.img = img
        self.labels = labels
        self.config = load_config.load_config('config.yaml')
        

    def __len__(self) :
        return self.img.size(0)

    def __getitem__(self, idx) :
        x = self.img[idx]
        y = self.labels[idx]

        return x, y

def _load_fashion_mnist (train=True) :
    fasion_mnist = datasets.FashionMNIST(
    root="../data",
    train=train,
    download=True,
    transform=transforms.Compose([
        ToTensor()
    ])
    )
    
    data = fasion_mnist.data
    target = fasion_mnist.targets

    return data, target

