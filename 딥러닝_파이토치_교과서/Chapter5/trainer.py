import os, time
from glob import glob
import torch
from tqdm import tqdm

class Train :
    def __init__(self,
                num_epoch,
                
                model,
                tr_loader,
                test_loader,
                criterion,
                optimizer,) :
        self.num_epoch = num_epoch

        self.model = model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = criterion
        self.optimizer = optimizer
        self.tr_loader = tr_loader
        self.test_loader = test_loader

    def train(self) :
        start = time.time()
        acc_hist = []
        loss_hist = []
        best_acc = 0.0

        print(f'Using Resource : {self.device}')

        print('Start Training...')
        for epoch in tqdm(range(1, self.num_epoch+1)) :
            


            running_loss = 0.0
            running_corrects = 0

            for imgs, labels in self.tr_loader :
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.model.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                

                _, preds = torch.max(outputs,1)

                running_loss += loss.item()*imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            
            epoch_loss = running_loss / len(self.tr_loader.dataset)
            epoch_acc = running_corrects.double() / len(self.tr_loader.dataset)

            print(f'EPOCH {epoch}/{self.num_epoch}')
            print(f'Loss : {epoch_loss:.4f}   Acc : {epoch_acc:.2f}')
            print('='*30)

            if epoch_acc > best_acc :
                best_acc = epoch_acc
                print(f'Saving Best Model at {epoch} Epoch')
                
                torch.save(self.model.state_dict(), './BEST_MODEL.pt')

            acc_hist.append(epoch_acc.item())
            loss_hist.append(epoch_loss)

        takes = time.time() - start

        print(f'Total Training Time : {takes//60}m')
        print(f'Best Accuracy : {best_acc*100:.2f}%')

        return acc_hist, loss_hist

    def eval(self) :
        start = time.time()

        acc_hist = []  
        best_acc = 0.0

        saved_model = './BEST_MODEL.pt'

        self.model.load_state_dict(torch.load(saved_model))
        self.model.eval()
        self.model.to(self.device)
        
        cor = 0
        total = 0

        for imgs, labels in self.test_loader :
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.no_grad() :
                outputs = self.model(imgs)

            _, preds = torch.max(outputs.data, 1)

            cor += (preds == labels).sum()

            total += len(labels)

        acc = cor * 100 / total
        
        print(f'TEST SCORE : {acc:.2f}%')

        return acc