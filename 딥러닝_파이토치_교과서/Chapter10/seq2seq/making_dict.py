from __future__ import unicode_literals, print_function, division
import torch

import pandas as pd

import os
import re
import random

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self):
        ## 단어의 index들이 들어갈 container 생성(초기화)
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  ## SOS와 EOS가 있으니 2, 점차 늘어날 예정

    def addSentence(self, sentence):
        for word in sentence.split():  ## 문장은 단어 단위로 분리 후 container에 추가
            self.addWord(word)

    def addWord(self, word):
        ## Container에 업데이트 하는 곳. 처음 들어오는 경우와 이미 들어왔던 경우로 나뉨
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1

        else:
            self.word2count[word] += 1


def normalizeString(df, lang):  # 들어오는 문장에 대한 가공
    sentence = df[lang].str.lower()  # 소문자 변환
    sentence = sentence.str.replace(
        "[^A-Za-z\s]+", " "
    )  # a-z, A-Z, ..., ? ,! 를 제외하고는 모두 공백으로 바꿈
    sentence = sentence.str.normalize("NFD")  # 유니코드 정규화 -> 공부!
    sentence = sentence.str.encode("ascii", errors="ignore").str.decode("utf-8")
    return sentence


def read_sentence(df, lang1, lang2):
    sentence1 = normalizeString(df, lang1)
    sentence2 = normalizeString(df, lang2)
    return sentence1, sentence2


def read_file(loc, lang1, lang2):
    df = pd.read_csv(loc, delimiter="\t", header=None, names=[lang1, lang2])
    # delimiter : sep과 같은 기능
    # names : columns와 같은 기능

    return df


def process_data(lang1, lang2):
    df = read_file(r"D:\fra.txt", lang1, lang2)

    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    ## 위에서 만들어 놨던 Lang class로 초기화
    input_lang = Lang()
    output_lang = Lang()

    pairs = []
    for i in range(len(df)):
        if (
            len(sentence1[i].split(" ")) < MAX_LENGTH and len(sentence2[i].split(" "))
        ) < MAX_LENGTH:
            full = [sentence1[i], sentence2[i]]
            input_lang.addSentence(sentence1[i])
            output_lang.addSentence(sentence2[i])
            pairs.append(full)

    return input_lang, output_lang, pairs


## 여기부턴 들어온 문장을 Tensor로 바꿔주는 역할
def indexesFromSentence(lang, sentence):
    # 아까 만들어놨던 container의 index 정보를 기반으로 index 정보를 반환

    return [lang.word2index[word] for word in sentence.split(" ") if word != ""]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)  # 위의 함수로 index를 가져오고 index의 끝에 문장의 끝을 알리는 EOS 추가
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
