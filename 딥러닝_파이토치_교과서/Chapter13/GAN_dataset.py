import imageio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import matplotlib
matplotlib.style.use('ggplot')

from torchvision.utils import make_grid, save_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

BATCH_SIZE = 512

def return_dataloader():
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ) (0.5, )),
    ])

    train_dataset = datasets.MNIST(
        root="딥러닝_파이토치_교과서/Chapter13/data", train=True, transform=transforms, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )

    return train_loader

