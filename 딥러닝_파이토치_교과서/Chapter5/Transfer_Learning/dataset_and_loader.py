import os, time, copy, glob,  shutil

import torch
import torchvision
import torchvision.transforms as transforms


from torch.utils.data import Dataset, DataLoader

class DogCatDataset :
    def __init__ (self) :

        os.chdir('딥러닝_파이토치_교과서\Chapter5\Transfer_Learning')

        self.train_root = './train'
        self.test_root = './test'

        self.TRAIN_TRANSFORM = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.TEST_TRANSFORM = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def to_LOADER(self) :
        self.train_dataset = torchvision.datasets.ImageFolder(self.train_root, transform=self.TRAIN_TRANSFORM)
        self.test_dataset = torchvision.datasets.ImageFolder(self.test_root, transform=self.TEST_TRANSFORM)

        print('# of Train Images : ', len(self.train_dataset))
        print('# of Test Images', len(self.test_dataset))

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=True)

        return self.train_loader, self.test_loader

        '''
        dataset = DogCatDataset()
        tr_loader, test_loader = dataset.to_LOADER()
        '''