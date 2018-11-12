import collections
import random
import tensorflow as tf
import numpy as np


class OneHotRNN:
    def __init__(self, n_input, n_hidden, training_iters, filename):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.training_iters = training_iters
        self.data, self.lines, self.line_num = self.read_data(filename)
        self.dictionary, self.reverse_dictionary = self.build_dataset(self.data)
        self.vocab_size = len(self.dictionary)
        return

    def read_data(self, filename):
        data = []
        lines = []
        with open(filename, 'r') as f:
            line_num = 0
            for line in f.readlines():
                line_num += 1
                words = line.replace('\n', '')
                lines.append(words.split(' '))
                for word in words.split(' '):
                    data.append(word)
        return data, lines, line_num

    def build_dataset(self, words):
        count = collections.Counter(words).most_common()
        dictionary = dict()
        dictionary['UNK'] = 0
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary

    def build_rnn(self, x, weight, bias):
        x = tf.reshape(x, [-1, self.n_input])
        x = tf.split(x, self.n_input, 1)
        layer1 = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
        layer2 = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([layer1])
        outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weight) + bias

    def build_train_data(self, offset, line):
        # 输入字符序号
        output = np.zeros([self.vocab_size], dtype=float)
        if offset + self.n_input >= len(line):
            input_key = [self.dictionary[line[i]] for i in range(offset, len(line))]
            for i in range(len(line), offset + self.n_input):
                input_key.append(self.dictionary['UNK'])
            output[self.dictionary['UNK']] = 1.0
        else:
            input_key = [self.dictionary[line[i]] for i in range(offset, start_offset + self.n_input)]
            # 输入字符的下一个字符的概率为1,其余为0
            output[self.dictionary[self.data[offset + self.n_input]]] = 1.0

        return np.reshape(input_key, [-1, self.n_input, 1]), np.reshape(output, [1, -1])


rnn_lstm = OneHotRNN(5, 120, 50000, 'yyyy.txt')

weight = tf.Variable(tf.random_normal([rnn_lstm.n_hidden, rnn_lstm.vocab_size]), name='W')
bias = tf.Variable(tf.random_normal([rnn_lstm.vocab_size]), name='bias')

x = tf.placeholder(dtype=tf.float32, shape=[None, rnn_lstm.n_input, 1], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, rnn_lstm.vocab_size], name='y')

prediction = rnn_lstm.build_rnn(x, weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(cost)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    display_step = 1000
    loss_sum = 0
    acc_sum = 0
    for step in range(rnn_lstm.training_iters):
        start_offset = random.randint(0, rnn_lstm.n_input + 1)
        end_offset = start_offset + rnn_lstm.n_input + 1
        num = random.randint(0, rnn_lstm.line_num - 1)

        if num < rnn_lstm.line_num:
            line = rnn_lstm.lines[num]
            if start_offset > (len(line) - end_offset):
                start_offset = random.randint(0, rnn_lstm.n_input + 1)

            key, value = rnn_lstm.build_train_data(start_offset, line)
            _, acc, loss, pred = sess.run([optimizer, accuracy, cost, prediction], feed_dict={x: key, y: value})
            loss_sum += loss
            acc_sum += acc
            if step % display_step == 0:
                print('Iter: ', step, ', Average Loss: ', '{:.6f}'.format(loss_sum / display_step),
                      ', Average Accuracy: ',
                      '{:.2f}%'.format(acc_sum / display_step * 100))
                acc_sum = 0
                loss_sum = 0

                if start_offset + rnn_lstm.n_input >= len(line):
                    input_word = [line[i] for i in range(start_offset, len(line))]
                    for i in range(len(line), start_offset + rnn_lstm.n_input):
                        input_word.append('UNK')
                    output_word = 'UNK'
                else:
                    input_word = [line[i] for i in range(start_offset, start_offset + rnn_lstm.n_input)]
                    output_word = line[start_offset + rnn_lstm.n_input]
                pred_word = rnn_lstm.reverse_dictionary[int(sess.run(tf.argmax(pred, 1)))]
                print('input: [%s] - output: [%s] vs pred: [%s]', (input_word, output_word, pred_word))
            start_offset += (rnn_lstm.n_input + 1)
            if start_offset > (len(line) - end_offset):
                num += 1
    print("Optimization Finished!")

    while True:
        prompt = "%s words: " % rnn_lstm.n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != rnn_lstm.n_input:
            continue
        try:
            symbols_in_keys = [rnn_lstm.dictionary[words[i]] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, rnn_lstm.n_input, 1])
                onehot_pred = sess.run(prediction, feed_dict={x: keys})
                pred_index = int(sess.run(tf.argmax(onehot_pred, 1)))
                sentence = "%s %s" % (sentence, rnn_lstm.reverse_dictionary[pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(pred_index)
            print(sentence)
        except ImportError:
            print("Word not in dictionary")
