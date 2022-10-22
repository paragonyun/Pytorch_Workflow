from dataset import FashionMNIST
from dataloader import FashionDataLoader
from model import FashionCNN
from trainer import Train
from torchsummary import summary
import torch.nn as nn
import torch


## dataset 
dataset = FashionMNIST()
train_dataset, test_dataset = dataset.down_and_return_dataset()

## dataloader
loader = FashionDataLoader(train_dataset, test_dataset)
train_loader, test_loader = loader.loaders()

## model
model = FashionCNN()

## Define Train Parmas
NUM_EPOCHS = 10
LR = 0.01
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)

print(summary(model, (1,28,28)))

trainer = Train(num_epoch = NUM_EPOCHS,
                model = model,
                tr_loader = train_loader,
                test_loader = test_loader,
                criterion = CRITERION,
                optimizer = OPTIMIZER)

acc_hist, loss_hist = trainer.train()

test_score = trainer.eval()

