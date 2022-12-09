from DataSetLoader import *
from train import *
from utils import *

import torch.optim as optim

_, model = bert_tokenizer()

train_loader, val_loader, test_loader = return_dataloaders()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(
    model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer
)
