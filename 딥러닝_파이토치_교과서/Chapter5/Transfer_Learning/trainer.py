import os, time
from glob import glob
import torch
from tqdm import tqdm

class Trainer :
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
        print('Start Training...')
        start = time.time()
        acc_hist = []
        loss_hist = []
        best_acc = 0.0

        for epoch in range(1, self.num_epoch + 1) :
            running_loss = 0.0
            running_cor = 0

            for imgs, labels in tqdm(self.tr_loader) :
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.model.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(imgs)

                _, preds = torch.max(outputs, 1)

                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                running_cor += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(self.tr_loader.dataset)
            epoch_acc = running_cor.double() / len(self.tr_loader.dataset)

            print(f'EPOCH {epoch}')
            print(f'LOSS : {epoch_loss:.2f}\tAccuracy : {epoch_acc*100:.2f}%')

            if epoch_acc > best_acc :
                best_acc = epoch_acc
                print(f'\nBEST MODEL IS SAVED at {epoch} epoch')

                torch.save(self.model.state_dict(), './BEST_MODEL.pt')

            acc_hist.append(epoch_acc)
            loss_hist.append(epoch_loss)

        it_takes = time.time() - start
        print(f'It takes... {it_takes//60:.1f}m')
        print(f'BEST ACCURACY : {best_acc}')
        
        return acc_hist, loss_hist

    def eval(self) : 
        start = time.time()
        
        self.model.load_state_dict(torch.load('./BEST_MODEL.pt'))
        self.model.eval()
        self.model.to(self.device)

        running_cor = 0 

        for imgs, labels in self.test_loader :
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.no_grad() :
                outputs = self.model(imgs)

            _, preds = torch.max(outputs, 1)

            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

            running_cor += preds.eq(labels.cpu()).int().sum()

        test_score = running_cor.double() / len(self.test_loader.dataset)
        print(f'TEST ACCURACY : {test_score*100:.2f}%')

        return test_score