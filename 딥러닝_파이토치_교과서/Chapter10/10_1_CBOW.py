from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings("ignore")
import gensim
from gensim.models import Word2Vec

sample = open("./peter.txt", "r", encoding="utf-8")
s = sample.read()

f = s.replace("\n", " ")

data = []

for i in sent_tokenize(f):  ## sent_tokenize()로 문장 단위로 바뀜 -> 문장마다 수행
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)

model1 = Word2Vec(
    data,
    min_count=1,  # 단어의 빈도수 제한 (빈도가 적은 단어는 학습하지 않는다.)
    vector_size=100,  # 임베딩된 벡터의 차원 (N)
    window=5,  # Context의 윈도우 크기
    sg=0,
)  # 0 : CROW,  1 : Skip-gram

print(f"'peter'와 'wendy'의 코사인 유사도 : {model1.wv.similarity('peter', 'wendy')}")
print(f"'peter'와 'hook'의 코사인 유사도 : {model1.wv.similarity('peter', 'hook')}")
