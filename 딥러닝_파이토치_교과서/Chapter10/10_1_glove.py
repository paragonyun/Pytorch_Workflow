from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


glove_file = datapath("D:\glove.6B.100d.txt")
word2vec_glove_file = get_tmpfile("glove.6B.100d.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
similarities = model.most_similar("bill")
for word, sim in similarities:
    print(f"Word : {word}, 유사도 : {sim:.2f}")
