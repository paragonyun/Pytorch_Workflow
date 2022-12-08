import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim  # 인코더에서 사용할 입력층
        self.embbed_dim = embbed_dim  # 인코더에서 사용할 임베딩 층
        self.hidden_dim = hidden_dim  # 인코더에서 사용할 은닉층(previous_hidden)
        self.num_layers = num_layers  # 인코더에서 사용할 GRU 게층 수
        self.embedding = nn.Embedding(input_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden
