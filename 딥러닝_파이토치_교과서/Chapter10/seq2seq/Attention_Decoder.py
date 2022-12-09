import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 20


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(
            self.hidden_size * 2, self.max_length
        )  # input sequence와 길이가 같은 인코딩된 sequence를 반환하는 역할이므로 output은 max_length
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )  # bmm : 배치 행렬곱. 배치부분을 제외한 차원끼리의 행렬곱을 실시!
        # 즉, 이건 가중치와 인코더의 출력 벡터를 곱하겠다는 말. attn_applied는 특정 부분에 관한 정보를 포함!
