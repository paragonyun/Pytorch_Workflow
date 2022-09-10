from gettext import npgettext
import torch
import numpy as np
import dataset

class CustomTrainer() :
    def __init__(self, model, criterion, metric, optimizer, config, device,
                data_loader, logging_step, val_data_loader=None) :
        self.config = config
        self.device = device

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

        self.check_point_dir = config.save_dir

        self.logging_step = logging_step

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

        for epoch in range(self.start_epoch, self.epochs + 1) :
            self.model.train()
            
            for batch_idx, (data, label) in enumerate(self.data_loader) :
                data, label = data.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.logging_step == 0 :
                    print(f'EPOCH : {epoch} BATCH : {batch_idx}\n Train Loss : {loss.item():.2f}')

## train 파트 마저 작성하기