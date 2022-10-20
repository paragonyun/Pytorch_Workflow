'''
DataLoader
Data의 Batch를 생성해주는 클래스
학습을 시작하기 직전에 데이터의 변환작업을 맡는다.

보통 Tensor로 변환해주고 Batch로 묶는 게 주요 업무

'''
import torch
from torch.utils.data import Dataset, DataLoader

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

text = ['Happy', 'Sad', 'Amazing', 'Love']
labels = ['Positive','Negative', 'Positive', 'Positive']

MyDataset= CustomDataset(text, labels)

MyDataLoader = DataLoader(MyDataset, batch_size=3, shuffle=True)

for data in MyDataLoader :
    print(data)

'''
Data Loader의 주요 파라미터
sampler : 데이터를 어떻게 뽑을지 결정하는 기법
collate_fn : Variable Length(가변자)를 처리할 때 사용, 
            글자수가 다른 애들을 어떻게 처리할지(0으로 패딩) 
            정의할 수 있음
'''