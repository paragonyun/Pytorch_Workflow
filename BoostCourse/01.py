'''
논문은 결국 수많은 반복의 연속
Layer = Block처럼 조립하는 구조!

Transformer나 ResNet도 결국 작은 블록으로 구성되어 있는 큰 블록들로 구성되어 있음

여러 블록의 연속이다!

그런 블록을 위한 것이 바로 nn.Module
input, output, forward, backward(autograd) 4개를 정의!
parameter 정의
'''

'''
nn.Paramter로 직접 설정해주는 일은 없지만 (이미 torch에 구현됨)
알아는 두자
'''
from torch import nn, Tensor
import torch

class MyLinear(nn.Module) :
    def __init__(self, in_features, out_features, bias=True) :
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(
                        torch.randn(in_features, out_features)
        )

        self.bias = nn.Parameter(
                        torch.randn(out_features)
        )


    def forward(self, x : Tensor) :
        return x@self.weights + self.bias

x = torch.randn(5, 7)

linear = MyLinear(7, 12)
print(linear(x))

for value in linear.parameters() :
    print(value)