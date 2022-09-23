from turtle import forward
import torch.nn as nn
import numpy as np

from abc import abstractmethod
## 부모 클래스에 이 Decorator를 달아두면
## 상속 받은 자식 클래스는 Decorator가 달린 메소드는 무조건 구현해야됨
## 나중에 유지 보수할 때 구현 안 해놨다가 발생하는 사고들을 막기 위함


class CustomModel(nn.Module) :

    @abstractmethod
    def __init__(self, x) :
        self.conv_layer1 = nn.Sequential(
            ## fashion mnist는 흑백 이미지이다 => in_channels=1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels= 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels= 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            # nn.Flatten(), 이게 없어도 foward 에서 view로 구현
            nn.Linear(in_features=16*3*3, out_features=128, bias=True),
            nn.ReLu()
        )

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10, bias = True),
            nn.ReLu()
        )

        raise f"{__name__} 함수는 꼭 작성 되어야 합니다"

    @abstractmethod
    def forward(self, x) :
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = output.view(output.shape[0], -1)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output

