import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

class FashionMNIST :
    def __init__(self) :
        self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])

        self.test_transform = transforms.Compose([
                transforms.ToTensor()
                ])

    def down_and_return_dataset(self) :

        train_dataset = torchvision.datasets.FashionMNIST(
            root = './data', download=True, transform=self.train_transform
        )

        test_dataset = torchvision.datasets.FashionMNIST(
            root = './data', download=True, transform=self.test_transform
        )

        return train_dataset, test_dataset

    '''
    dataset = FashionMNIST()
    train_dataset, test_dataset = dataset.down_and_return_dataset()
    '''