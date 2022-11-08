from dataset import return_dataset
from LSTM import LSTM
import torch.nn as nn
from torch.optim import Adam


X_train, X_test, y_train, y_test = return_dataset()

NUM_EPOCH = 500
LR = 1e-4

INPUT_SIZE = 5 ## column 수
HIDDEN_SIZE = 2 ## 안의 뉴런 수
NUM_LAYERS = 1 ## LSTM 계층 수

NUM_CLS = 1
model = LSTM(num_cls=NUM_CLS, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, seq_length=X_train.shape[1])

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LR)

print('🎆학습을 시작합니다🎆')
loss_history = []
for epoch in range(NUM_EPOCH) :
    outputs = model.forward(X_train)
    optimizer.zero_grad()

    loss = criterion(outputs, y_train)
    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    if epoch%500 ==0 :
        print(f'EPOCH : {epoch}, LOSS : {loss.item()}')
