"""
https://fasttext.cc/docs/en/pretrained-vectors.html
에서 다운 뒤, 해당 파일 사용
"""

from __future__ import print_function
from gensim.models import KeyedVectors

print("단어 집합 학습중...")
model_kr = KeyedVectors.load_word2vec_format("D:\wiki.ko.vec")

target_word = "노력"

for similar_word in model_kr.similar_by_word(target_word):
    print(f"Word : {similar_word[0]}, 유사도 : {similar_word[1]:.2f}")


## 특정 단어와 긍정적인 혹은 부정적인 단어 찾기
similarities = model_kr.most_similar(positive=["동물", "육식동물"], negative=["사람"])
for word, sim in similarities:
    print(f"Word : {word}, 유사도 : {sim:.2f}")
