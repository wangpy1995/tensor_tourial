import collections
import jieba
import numpy as np
import tensorflow as tf


class Word2Vec:
    def __init__(self, filename, embedding_size):
        self.embedding_size = embedding_size
        self.corpus = self.read_data(filename)
        self.dictionary, self.reverse_dictionary = self.build_data()
        self.vocab_size = len(self.corpus)
        return

    def read_data(self, filename):
        with open(filename) as f:
            words = f.read().replace('\n', '').replace(' ', '')
            corpus = jieba.cut(words)
        return corpus

    def build_data(self):
        count = [['UNK', -1]]
        count.extend(collections.Counter(self.corpus).most_common())
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary

    def get_output_word(self, input_batch, idx, skip_window):
        target_window = np.random.randint(1, skip_window + 1)
        # 这里要考虑input word前面单词不够的情况
        start_point = idx - target_window if (idx - target_window) > 0 else 0
        end_point = idx + target_window
        # output words(即窗口中的上下文单词)
        targets = set(input_batch[start_point: idx] + input_batch[idx + 1: end_point + 1])
        return list(targets)

    def build_train_data(self, offset, batch_size, skip_window):
        assert offset - skip_window >= 0
        input_key = [self.dictionary[self.corpus[i]] for i in range(offset, offset + batch_size)]
        x = []
        y = []
        # 输入input_key每个字符相邻字符id
        for idx in range(len(input_key)):
            output = self.get_output_word(input_key, idx, skip_window)
            input_x = [input_key[idx] * len(output)]
            x.append(input_x)
            y.append(output)
        return x, y


w2v = Word2Vec('train_data_zh', 200)

x = tf.placeholder(tf.float32, shape=None, name='x')
y = tf.placeholder(tf.float32, shape=[None, None], name='y')

embedding = tf.Variable(tf.random_uniform([w2v.vocab_size, w2v.embedding_size], -1, 1), name='embedding')
embed = tf.nn.embedding_lookup(embedding, x)

biases = tf.Variable(tf.zeros(w2v.vocab_size), name='bias')
weight = tf.Variable(tf.truncated_normal([w2v.vocab_size, w2v.embedding_size], stddev=0.1), name='W')
loss = tf.nn.nce_loss(weight, biases, y, x, 100, w2v.vocab_size)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(cost)


