'''
model.save()
- 모델의 형태 혹은 파라미터를 저장할 수 있음.
- 최선의 결과만 선택할 수 있음
- 외부 연구자와 공유하여 학습 재연성을 향상시킬 수 있음.

torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model.pt"))

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pt")))
'''

from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchsummary import summary

class MyModel(nn.Module) :
    def __init__(self) :
        super(MyModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 0),
            nn.Conv2d(3, 3, 3, 3, 3, 3),
            nn.Conv2d(3, 3, 3, 3, 3, 3),
            nn.Conv2d(3, 3, 3, 3, 3, 3),
            )

        self.drop_out = nn.Dropout(0.3)
        self.fc_layer1 = nn.Linear(3, 10)
        self.fc_layer2 = nn.Linear(10, 1)

    def forward(self, x) :
        output = self.layer1(x)
        
        output = output.view(output.size(0), -1)

        output = self.drop_out(output)
        output = self.fc_layer1(output)
        output = self.fc_layer2(output)

        return output

MODEL_PATH = './MODEL_CHECKPOINT'

model = MyModel()

summary(model, (3, 224, 224))

torch.save(model.state_dict(), os.path.join(MODEL_PATH, "model.pt"))

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "model.pt")))







