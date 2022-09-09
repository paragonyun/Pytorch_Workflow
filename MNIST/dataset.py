from torchvision.datasets import MNIST
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import torch

from torch.utils.data import Dataset


import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset) :
    def __init__(self, img, labels) :
        self.img = img
        self.labels = labels
        

    def __len__(self) :
        return self.img.size(0)

    def __getitem__(self, idx) :
        x = self.img[idx]
        y = self.labels[idx]

        return x, y

def load_fashion_mnist (train=True) :
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

