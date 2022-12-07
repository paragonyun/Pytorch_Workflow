from gensim.test.utils import common_texts
from gensim.models import FastText

model = FastText("./peter.txt", vector_size=4, window=3, min_count=1, epochs=10)

print(f"'peter'와 'wendy'의 코사인 유사도 : {model.wv.similarity('peter', 'wendy')}")
print(f"'peter'와 'hook'의 코사인 유사도 : {model.wv.similarity('peter', 'hook')}")
