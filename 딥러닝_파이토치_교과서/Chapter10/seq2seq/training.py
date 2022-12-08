import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm

from making_dict import tensorsFromPair, tensorFromSentence

teacher_force_ratio = 0.5
MAX_LENGTH = 20
EOS_token = 1


def Model(
    model, input_tensor, target_tensor, model_optimizer, criterion
):  # 모델의 loss 계산하는 부분 정의
    model_optimizer.zero_grad()
    input_lengt = input_tensor.size()

    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])  # index로 계산..!

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter

    return epoch_loss


def trainModel(model, input_lang, output_lang, pairs, num_iteration=20000):
    print("🚀Start Training🚀")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()  # CE와 똑같은데, 이건 마지막에 softmax가 없음! 따로 명시해줘야됨
    total_loss_iterations = 0

    training_pairs = [
        tensorsFromPair(input_lang, output_lang, random.choice(pairs))
        for i in range(num_iteration)
    ]  # 반복하 ㄹ대마다 pair의 tensor list 반환 (data, target)

    for iter in tqdm(range(1, num_iteration + 1)):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = Model(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss

        if iter % 5000 == 0:
            avg_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print(f"EPOCH : {iter}\t LOSS : {avg_loss:3f}")

    torch.save(model.state_dict(), "./training.pt")

    return model


def evalueate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    print("✨Start Evaluation✨")
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0])
        output_tensor = tensorFromSentence(output_lang, sentences[1])

        decoded_words = []
        output = model(input_tensor, output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)  # 각 출력에서 가장 높은 값과 인덱스 반환

            if topi[0].item() == EOS_token:  # 끝났다고 판단하면...
                decoded_words.append("<EOS>")
                break
            else:  # 계속 반복
                decoded_words.append(
                    output_lang.index2word[topi[0].item()]
                )  # 가장 확률이 높은 단어의 index로 word를 찾아서 append 함

    return decoded_words


def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)  # 랜덤한 문장을 가져옴
        print(f"Input : {pair[0]}")
        print(f"Output : {pair[1]}")
        output_words = evalueate(model, input_lang, output_lang, pair)
        output_sentence = " ".join(output_words)
        print(f"Predicted : {output_sentence}")
