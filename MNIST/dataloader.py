from ast import Sub
import numpy as np

## 상속 받을 거 import
from torch.utils.data import DataLoader

## validation ratio를 나눌 때 random으로 sample 하게 도와줄 도구
from torch.utils.data.sampler import SubsetRandomSampler

import dataset

class CustomDataLoader(DataLoader) :
    def __init__ (self, dataset, batch_size,  val_ratio, train=True) :
        self.val_ratio = val_ratio

        self.dataset = dataset
        self.batch_size = batch_size

        self.train = train


        self.train_sample, self.val_sample = self._spliter(self.val_ratio)
        


    def _spliter(self, ratio) :
        if ratio == 0 :
            return None, None

        ## 최대 길이 만큼의 값을 뽑아냄
        idx_length = np.arange(len(self.dataset))

        ## 전체 데이터셋 * ratio를 validation 수로 지정
        val_length = int(len(self.dataset) * ratio)

        ## 비율 나누기
        val_idx = idx_length[:val_length]
        train_idx = np.delete(idx_length, np.arange(0, val_length))

        ## sampling !!
        train_sampling = SubsetRandomSampler(train_idx)
        val_sampling = SubsetRandomSampler(val_idx)

        return train_sampling, val_sampling
    
    
    ## validation 을 나누면~
    def split_validation(self) :
        if self.val_sample is None :
            return None
        elif self.train :            
            ## sampler 파라미터에 val을 넣고 나머지 파라미터를 넣어줌
            train_loader = DataLoader(self.dataset ,sampler= self.train_sample, batch_size= self.batch_size)
            val_loader =  DataLoader(self.dataset,sampler = self.val_sample, batch_size= self.batch_size)
            return train_loader, val_loader

        else : ## test dataset인 경우
            return DataLoader(**self.init_kwargs)
        ''' 
        나중에 train.py에서 
        from dataloader import split_validation
        dl = DataLoader(train_dataset, 32, 0.2)
        train_loader , val_loader = dl.split_validation()
        해줄 생각
        '''
