'''
다른 데이터셋으로 만든 모델을 현재 모델에 적용
그렇게 학습된 모델의 일부만 변경해서 학습 수행

모델의 일부분을 Frozen 시킴
'''

import torch
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg = models.vgg16(pretrained=True).to(device)
print(vgg) # 모델의 구조 출력

## Layer의 이름과 구조 확인
for name, layer in vgg.named_moduls() :
    print(name, layer)


## 마지막 계층을 추가하는 방법
vgg.fc = torch.nn.Linear(1000, 1) ## 마지막 Layer에 FC Layer 추가
vgg.cuda()

# or

# vgg.층이름.레이어번호 = torch.nn.Linear(xxx, x)

import torch.nn as nn

class MyNewNet(nn.Module) :
    def __init__(self) :
        super(MyNewNet, self).__0init__()

        self.vgg19 = models.vgg19(pretrained=True)
        self.linear_layer = nn.Linear(1000,1) ## 이미지넷은 마지막에 출력이 1000, 이니까 이걸 1로 바꿔주는 거 만듦

    def forward(self, x) :
        x = self.vgg19(x)
        return self.linear_layer(x)

model = MyNewNet().to(device)


for param in model.parameters() :
    param.requires_grad = False ## 전체 일단 Frozen 시키기 

for param in model.linear_layer.parameters() : 
    param.requires_grad = True ## 얘는 그 와중에 다시 풀기









