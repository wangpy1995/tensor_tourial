# 读取模型

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
saver = tf.train.import_meta_graph("MNIST_model/m.ckpt-1.meta")
model_file = tf.train.latest_checkpoint("MNIST_model/")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_file)
    graph = tf.get_default_graph()

    x = tf.placeholder("float", [None, 784])  # 图片像素 28*28

    W = graph.get_tensor_by_name('W:0')
    b = graph.get_tensor_by_name('bias:0')
    y = tf.nn.softmax(tf.matmul(x, W) + b)  # 模型

    y_ = tf.placeholder("float", [None, 10])

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
