import nltk

# nltk.download('punkt')
string1 = "my favorite subject is AI"
string2 = "my favorite subject is AI, economics and business"

print(nltk.word_tokenize(string1))  # ['my', 'favorite', 'subject', 'is', 'AI']
print(
    nltk.word_tokenize(string2)
)  # ['my', 'favorite', 'subject', 'is', 'AI', ',', 'economics', 'and', 'business']

from konlpy.tag import Komoran

komoran = Komoran()

##형태소로 변환
print(komoran.morphs("딥러닝이 쉽나요? 어렵나요?"))  # ['딥러닝이', '쉽', '나요', '?', '어렵', '나요', '?']

## 형태소 + 품사 태깅
print(
    komoran.pos("딥러닝이 쉽나요? 어렵나요?")
)  # [('딥러닝이', 'NA'), ('쉽', 'VA'), ('나요', 'EF'), ('?', 'SF'), ('어렵', 'VA'), ('나요', 'EF'), ('?', 'SF')]
