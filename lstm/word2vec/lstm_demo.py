import tensorflow as tf
import word2vec.word


def RNN(x, weights, biases, n_input, n_hidden):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input, 1)
    rnn_cell = tf.rnn.rnn_cell.BasicLSTMCell(n_hidden)
    outputs, state = tf.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


dictionary = word2vec.word.dictionary
vocab_size = len(dictionary)
n_input = 3
n_hidden = 512
weights = word2vec.word.nce_weights
biases = word2vec.word.nce_biases

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
