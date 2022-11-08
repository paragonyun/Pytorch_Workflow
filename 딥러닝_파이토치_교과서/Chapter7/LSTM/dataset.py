import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
import torch

def return_dataset() :

    data = pd.read_csv('딥러닝_파이토치_교과서\Chapter7\LSTM\SBUX.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    # date를 index로 사용
    data.set_index('Date', inplace=True)

    # 데이터 형식 변경
    data['Volume'] = data['Volume'].astype(float)

    # 라벨과 Feature 분리
    X = data.iloc[:, :-1]
    y = data.iloc[:, 5:6]

    # Scaling
    ms = MinMaxScaler()
    ss = StandardScaler()

    X_ss = ss.fit_transform(X)
    y_ms = ms.fit_transform(y)

    X_train = X_ss[:-200, :]
    X_test = X_ss[-200:, :]
    y_train = y_ms[:-200, :]
    y_test = y_ms[-200:, :]

    print('Shape of Dataset')
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    ## Make it Tensor
    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))


    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    ## LSTM 입력 형태를 맞춰주기 위함
    X_train_tensors_f = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_f = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))


    return X_train_tensors_f, X_test_tensors_f, y_train_tensors, y_test_tensors