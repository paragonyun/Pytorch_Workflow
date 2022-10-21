import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
import requests

mnist_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0, ))
                                    ])

ROOT = '../data/MNIST_DATASET'
train_dataset = MNIST(root=ROOT, transform=mnist_transform, train=True, download=True)
val_dataset = MNIST(root=ROOT, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(root=ROOT, transform=mnist_transform, train=False, download=True)

