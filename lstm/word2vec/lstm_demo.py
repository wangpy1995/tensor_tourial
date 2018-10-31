import os
import numpy as np

import nltk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

# 读入文本
raw_text = ""
raw_text += open("lstm_train.txt", errors="ignore").read() + "\n\n"
# row_test=open("../input/Winston_Churchil.txt").read()
raw_text = raw_text.lower()
# sentensor = nltk.data.load("tokenizers/punkt/english.pickle")
# sents = sentensor.tokenize(raw_text)
sents = nltk.sent_tokenize(raw_text)
corpus = []
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))

print(len(corpus))
print(corpus[:3])

if not os.path.exists('w2v_model'):
    w2v_model = Word2Vec(corpus, size=200, window=5, min_count=5, workers=4)
    w2v_model.save('w2v_model')
else:
    import gensim

    w2v_model = gensim.models.word2vec.Word2Vec.load('w2v_model')

w2v_model["office"]

raw_input = [item for sublist in corpus for item in sublist]
len(raw_input)
raw_input[12]

text_stream = []
vocab = w2v_model.wv.vocab
for word in raw_input:
    if word in vocab:
        text_stream.append(word)
len(text_stream)

seq_length = 10
x = []
y = []
for i in range(0, len(text_stream) - seq_length):
    given = text_stream[i:i + seq_length]
    predict = text_stream[i + seq_length]
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])
print(x[10])
print(y[10])

print(len(x))
print(len(y))
print(len(x[12]))
print(len(x[12][0]))
print(len(y[12]))

# 行数未知，用-1表示
x = np.reshape(x, (-1, seq_length, 200))
y = np.reshape(y, (-1, 200))

if not os.path.exists('lstm'):
    model = Sequential()
    model.add(LSTM(512, dropout=0.2, input_shape=(seq_length, 200), return_sequences=True))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation="softmax"))
    model.compile(loss="mse", optimizer="Adam")

    # 跑模型
    model.fit(x, y, epochs=150, batch_size=4096)
    model.save('lstm')
else:
    import keras

    model = keras.models.load_model('lstm')


# 预测
def predict_next(input_array):
    x = np.reshape(input_array, (-1, seq_length, 200))
    y = model.predict(x)
    return y


def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[(len(input_stream) - seq_length):]:
        if w2v_model.wv.__contains__(word):
            res.append(w2v_model[word])
        else:
            res.append(w2v_model["job"])
    return res


def y_to_word(y):
    word = w2v_model.most_similar(positive=y, topn=1)
    return word


def generate_article(init, rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(string_to_index(in_string)))
        in_string += " " + n[0][0]
    return in_string


init = 'Language Models allow us to measure how likely a answer is, which is an important for Machine'
article = generate_article(init)
print(article)
