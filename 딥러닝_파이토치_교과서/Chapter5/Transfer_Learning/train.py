from dataset_and_loader import DogCatDataset
from model import ResNet18
from trainer import Trainer
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

dataset = DogCatDataset()
train_loader, test_loader = dataset.to_LOADER()

model = ResNet18()
model = model.run()

## Define Train Parmas
NUM_EPOCHS = 10
LR = 0.01
CRITERION = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)

trainer = Trainer(num_epoch=NUM_EPOCHS,
                model=model,
                tr_loader=train_loader,
                test_loader=test_loader,
                criterion=CRITERION,
                optimizer=OPTIMIZER)

acc_hist, loss_hist = trainer.train()

score = trainer.eval()

plt.plot(acc_hist)
plt.plot(loss_hist)


