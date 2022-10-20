from re import L
from torchvision.datasets import VisionDataset
from typing import Any, Callable, List, Optional, Tuple
import os
from tqdm import tqdm
import sys
from pathlib import Path
import requests
from skimage import io, transform
import matplotlib.pyplot as plt
import tarfile

class NotMNIST(VisionDataset) :
    resource_url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz'

    def __init__ (
                self, 
                root,
                train=True,
                transform=None,
                target_transform=None,
                download=False ) :

        super(NotMNIST, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        
        if not self._check_exists() :
            self.download()

        if download :
            self.download()

        self.data, self.targets = self._load_data()

    
    def __len__(self) :
        return len(self.data)

    def __getitem__(self, index) :
        image_name = self.data[index]
        image = io.imread(image_name)
        label = self.targets[index]
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_data(self) :
        filepath = self.image_folder 
        data = []
        targets = []

        for target in os.listdir(filepath) :
            filenames = [os.path.abspath(
                os.path.join(filepath, target, x)) for x in os.listdir(
                    os.path.join(filepath, target))]
            
            targets.extend([target] * len(filenames))
            data.extend(filenames)

        return data, targets
            
    @property
    def raw_folder(self) :
        return os.path.join(self.root, self.__class__.__name__ ,'raw')

    @property
    def image_folder(self) :
        return os.path.join(self.root, 'notMNIST_larget')


    def download(self) :
        os.makedirs(self.raw_folder, exist_ok = True)
        fname = self.resource_url.split('/')[-1]
        chunk_size = 1024

        print(requests.head(self.resource_url).headers)

        filesize = int(requests.head(self.resource_url).headers["Content-Length"])

        with requests.get(self.resource_url, stream=True) as r, open(
            os.path.join(self.raw_folder, fname), 'wb') as f, tqdm(
                unit='8',
                unit_scale=True,
                unit_divisor=1024,
                total=filesize,
                file=sys.stdout,
                desc=fname
            ) as progress :
            for chunk in r.iter_content(chunk_size=chunk_size) :
                datasize = f.write(chunk)
                progress.update(datasize)
        
        self._extract_file(os.path.join(self.raw_folder, fname), target_path=self.root)
    

    def _extract_file(self, fname, target_path) :
        if fname.endswith('tar.gz') :
            tag = 'r:gz'
        elif fname.endswith('tar') :
            tag = 'r:'
        
        tar = tarfile.open(fname, tag)
        tar.extractall(path=target_path)
        tar.close()

    def _check_exists(self) :
        return os.path.exists(self.raw_folder)

dataset = NotMNIST('data', download=True)

###############################################

import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means=[0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
])

dataset_loader = torch.utils.DataLoader(dataset, batch_size=128, shuffle=True)



