import torch.nn as nn
import torch
from torch.autograd import Variable

class LSTM(nn.Module) :
    def __init__(self, num_cls, input_size, hidden_size, num_layers, seq_length) :
        super(LSTM, self).__init__()

        self.num_cls = num_cls
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        ## 모델 정의
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fin_fc = nn.Linear(128, num_cls)

        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x) :
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        hn = hn.view(-1, self.hidden_size)

        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fin_fc(out)
        return out

    