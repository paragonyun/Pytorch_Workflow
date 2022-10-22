import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class FashionCNN(nn.Module) :
    def __init__(self) :
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, 
                        kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                        kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.dropout = nn.Dropout2d(0.3)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x) :
        output = self.layer1(x)
        output = self.layer2(output)
        
        output = output.view(output.size(0), -1)
        
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output