from Decoder import Decoder
from Encoder import Encoder

import torch.nn as nn
import torch

import random

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20
device = "cuda" if torch.cuda.is_available() else "cpu"


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LANTGH=MAX_LENGTH):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):
        input_length = input_lang.size(0)
        batch_size = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(
            self.device
        )  ## 출력값을 저장하기 위한 공간을 만듦

        for i in range(input_length):
            encoder_ouptut, encoder_hidden = self.encoder(input_lang[i])  ## 모든 던어를 인코딩!

        decoder_hidden = encoder_hidden.to(
            device
        )  # encoder에서 나온 hidden Layer를 decoder의 hidden 으로 사용한다.
        decoder_input = torch.tensor(
            [SOS_token], device=device
        )  # Decoder에서 나온 첫번재 예측 단어 맨 앞에 SOS 추가

        for t in range(target_length):  # 현재 단어에서 출력단어를 예측하는 지점
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = (
                random.random() < teacher_forcing_ratio
            )  # 0.5를 넘냐 안 넘냐로 True False가 나옴
            topv, topi = decoder_output.topk(
                1
            )  # topk : 큰 순서대로 n개씩 뽑아옵니다. value 와 index를 반환하는데 우린 index만 씀
            input = (
                output_lang[t] if teacher_force else topi
            )  # teacher_force가 True면 input은 t 시점의 output이 들어가고 아니면 topi가 들어갑니다.
            if (
                teacher_force == False and input.item() == EOS_token
            ):  # teacher_foce가 False고 topi의 결과가 EOS_token이면 자체적인 예측값을 다음 입력으로 사용합니다.
                break

        return outputs
