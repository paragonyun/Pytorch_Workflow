import numpy as np
import torch

class EarlyStopping :
    def __init__(self,path:str,  patience=5, verbose=False, delta=0) :

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model) :
        score = -val_loss
        
        if self.best_score is None :
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta : 
            self.counter += 1
            print(f'Early Stoppint Count : {self.counter} / {self.patience}')
            if self.counter >= self.patience :
                self.early_stop = True

        else : 
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0 

    def save_checkpoint(self, val_loss, model) :
        if self.verbose :
            print('Validation Loss is Decreaded!!')
            print(f'{self.val_loss_min:.4f} --> {val_loss:.4f} | Model is Saved')
        
        torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss        