from seq2seq import Seq2Seq
from making_dict import *
from training import *
from Encoder import Encoder
from Decoder import Decoder

import random

device = "cuda" if torch.cuda.is_available() else "cpu"

lang1 = "eng"
lang2 = "fra"

input_lang, output_lang, pairs = process_data(lang1, lang2)

randomize = random.choice(pairs)
print(f"Random Sentence : {randomize}")

input_size = (
    input_lang.n_words
)  ## 여기서 input_lang은 class 이므로 내부의 n_words 변수를 가지고 있음!!! 와 대박..
output_size = output_lang.n_words
print(f"Input : {input_size}, Ouput : {output_size}")

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 75000

encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

model = Seq2Seq(encoder, decoder, device).to(device)

print(encoder)
print(decoder)

model = trainModel(model, input_lang, output_lang, pairs, num_iteration)
