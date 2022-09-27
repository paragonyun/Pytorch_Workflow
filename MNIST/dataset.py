from torchvision.datasets import MNIST
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import load_config

import torch

from torch.utils.data import Dataset


import matplotlib.pyplot as plt


class CustomDataset(Dataset) :
    def __init__(self, img=None, labels=None, config=None) :
        self.img = img
        self.labels = labels
        self.config = load_config.load_config('./config.yaml')
        

    def __len__(self) :
        return self.img.size(0)

    def __getitem__(self, idx) :
        x = self.img[idx]
        y = self.labels[idx]

        return x, y

    def _load_fashion_mnist (self) :
        
        train_dataset = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=transforms.Compose([
            ToTensor()
        ])
        )
        
        test_dataset = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=transforms.Compose([
            ToTensor()
        ])
        )

    


        return train_dataset, test_dataset

