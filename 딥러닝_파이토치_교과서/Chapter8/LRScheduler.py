import torch

class LRScheduler :
    def __init__ (self, optimizer, patience=5, min_lr=1e-6, factor=0.5) :
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', ## 학습률을 언제 조절할 지 결정하는 기준이 되는 값 // loss 기준이면 'min', accuracy 기준이면 'max'를 이용합니다.
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss) : 
        self.lr_scheduler.step(val_loss)