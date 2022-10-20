'''
모델에 데이터를 먹이는 방법

Dataset
__init__ : 시작할 때 어떻게 불러올지
__len__ : 길이를 어떻게?
__getitem : Map-style 정의 = 하나의 데이터를 불러올 때 어떤 식으로 반환해올지

Dataloader
데이터를 정의한 다음에, 어떻게 전달해줄지를 결정
batch나 shuffle 여부 결정
'''

'''
DATASET
데이터 입력 형태를 정의하는 클래스
-> 데이터 입력 방식의 표준화
-> 이미지, 텍스트, 오디오 등에 따라 입력 정의가 다 다름

--- 주의할 점 ---
데이터의 형태마다 각 함수를 다르게 정의해야 한다.
모든 걸 생성시점에 굳이 처리할 필요는 없고 Tensor로의 변환같은 건 학습할 때
CPU가 Tensor로 변환하고 GPU는 학습만 시킴

표준화된 처리방법이 있다면 후속 연구자들에게 좋은 자료가 됨

허깅페이스같은 라이브러리 있는데, 그거 쓰면 좋습니당
!! 허깅페이스 / Fast AI 꼭 공부하기 !!
'''

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset) :
    def __init__(self, text, labels) : ## 초기 데이터를 어떻게 불러올지 결정
        self.labels = labels            ## 데이터 폴더의 디렉토리등을 정의해줌 
        self.data = text

    def __len__(self) :                 ## 데이터의 전체 길이를 반환
        return len(self.labels)

    def __getitem__(self, idx)  :       ## index를 줬을 때, 어떻게 반환해줄지 결정
        label = self.labels[idx]
        text = self.data[idx]
        sample = {'Text' : text,
                    'Class' : label}    ## 보통은 dict로 반환하긴 함
        return sample

text = ['Happy', 'Sad']
labels = ['Positive','Negative']

MyDataset= CustomDataset(text, labels)





