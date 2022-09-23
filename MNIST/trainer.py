from gettext import npgettext
import torch
import numpy as np
import dataset

import load_config

import dataloader
import model

class CustomTrainer() :
    def __init__(self, model, criterion, metric, optimizer, config, device,
                data_loader, logging_step, val_data_loader=None) :
        self.config = config
        self.device = self._get_device()

        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer['monitor']

        self.data_loader = data_loader


        self.start_epoch = 1

        self.check_point_dir = cfg_trainer['save_dir']

        self.logging_step = cfg_trainer['logging_step']

    def _get_device(self) :
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('학습 자원 : ', device)
        return device

    def _save_checkpoint(self, epoch) :
        model_name = type(self.model).__name__

        state = {
            'arch' : model_name,
            'epoch' : epoch,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'config' : self.config
        }

        filename = str(self.check_point_dir / f'checkpoint : epoch{epoch}.pth')
        torch.save(state, filename)
        print(f'EPOCH : {epoch}에서 모델 저장 완료')


    def train(self) :
        early_stop_to = 0

        print('Start Training...')

        for epoch in range(self.start_epoch, self.epochs + 1) :
            self.model.train()

            train_loss, train_acc = 0, 0
            
            for batch_idx, (data, label) in enumerate(self.data_loader) :
                data, label = data.to(self.device), label.to(self.device)

                y_pred = model(data)

                loss = self.criterion(y_pred, label)
                train_loss += loss.item()

                self.optimizer.zero_grad()

                loss.backward()
                
                self.optimizer.step()

                y_pred_prob = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                train_acc += (y_pred_prob == label).sum().item()/len(y_pred)

                ## loss와 정확도 계산
                train_loss = train_loss / len(dataloader)
                train_acc = train_acc / len(dataloader)

                if batch_idx % self.logging_step == 0 :
                    print(f'EPOCH : {epoch} BATCH : {batch_idx}\n Train Loss : {train_loss:.2f} | Train Acc : {train_acc:.2f}')

    # def test(self) :
        ''' # TODO
        1. 일단 validate 생각하지 말고 train_test 만 일단 구현
        2. 그 이후 validate 구현
        3. 그 이후 early stopping 구현
        4. 여기까지 되면 NLP데이터로도 해보기
        '''