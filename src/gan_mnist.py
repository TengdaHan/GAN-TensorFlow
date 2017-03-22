# tf rd test - mnist GAN
import tensorflow as tf
import numpy as np
import cv2
import os


def conv(x, w, b, stride, name):
    with tf.name_scope('conv'):
        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        return tf.nn.conv2d(x,
                           filter=w,
                           strides=[1, stride, stride, 1],
                           padding='SAME',
                           name=name) + b


def deconv(x, w, b, shape, stride, name):
    with tf.name_scope('deconv'):
        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        return tf.nn.conv2d_transpose(x,
                                       filter=w,
                                       output_shape=shape,
                                       strides=[1, stride, stride, 1],
                                       padding='SAME',
                                       name=name) + b


def read_data():
    # read mnist data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    return mnist


def discriminator(X, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        K = 4
        M = 8
        # W1 = tf.Variable(tf.truncated_normal([3, 3, 1, K], stddev=0.1))
        # B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        # W2 = tf.Variable(tf.truncated_normal([3, 3, K, M], stddev=0.1))
        W2 = tf.Variable(tf.truncated_normal([3, 3, 1, M], stddev=0.1))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
        W3 = tf.Variable(tf.truncated_normal([7*7*M, 100], stddev=0.1))
        B3 = tf.Variable(tf.constant(0.1, tf.float32, [100]))
        W4 = tf.Variable(tf.truncated_normal([100, 1], stddev=0.1))
        B4 = tf.Variable(tf.constant(0.1, tf.float32, [1]))

        # conv1 = conv(X, W1, B1, stride=2, name='conv1')
        # bn1 = tf.nn.batch_normalization(conv1, 0, 0.1,
        #                                offset=None,
        #                                scale=None,
        #                                variance_epsilon=1e-5)
        conv2 = conv(X, W2, B2, stride=2, name='conv2')
        bn2 = tf.nn.batch_normalization(conv2, 0, 0.1,
                                       offset=None,
                                       scale=None,
                                       variance_epsilon=1e-5)
        flat = tf.reshape(tf.nn.relu(bn2), [-1, 7*7*M], name='flat')
        dense = tf.nn.relu(tf.matmul(flat, W3) + B3, name='dense')
        logits = tf.matmul(dense, W4) + B4
        return tf.nn.sigmoid(logits), logits


def generator(X, batch_size):
    with tf.variable_scope('generator'):
        K = 4
        M = 2

        W1 = tf.Variable(tf.truncated_normal([10, 7*7*K], stddev=0.1))
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [7*7*K]))

        W2 = tf.Variable(tf.truncated_normal([2, 2, M, K], stddev=0.1))
        B2 = tf.Variable(tf.constant(1, tf.float32, [M]))
        S2 = tf.Variable(tf.constant([batch_size, 14, 14, M], tf.int32))  # 14

        W3 = tf.Variable(tf.truncated_normal([2, 2, 1, M], stddev=0.1))
        B3 = tf.Variable(tf.constant(0.1, tf.float32, [1]))
        S3 = tf.Variable(tf.constant([batch_size, 28, 28, 1], tf.int32))  # 28

        XX = tf.nn.relu(tf.matmul(X, W1) + B1)
        XX_reshape = tf.reshape(XX, [batch_size, 7, 7, K])
        deconv1 = deconv(XX_reshape, W2, B2, shape=S2, stride=2, name='deconv1')
        bn1 = tf.nn.batch_normalization(deconv1, 0, 0.1, 
        	offset=None, 
        	scale=None, 
        	variance_epsilon=1e-5)
        deconv2 = deconv(tf.nn.relu(bn1), W3, B3, shape=S3, stride=2, name='deconv2')
        bn2 = tf.nn.batch_normalization(tf.nn.relu(deconv2), 0, 0.1, 
        	offset=None, 
        	scale=None, 
        	variance_epsilon=1e-5)
        return tf.nn.tanh(bn2)


def train(batch_size=100):
    mnist = read_data()

    # real_label = np.hstack((np.zeros([batch_size,1]), np.ones([batch_size,1])))
    # fake_label = np.hstack((np.ones([batch_size, 1]), np.zeros([batch_size, 1])))

    # Raw image
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    tf.summary.image('raw image', X, 3)
    # # correct label
    # Y_real = tf.placeholder(tf.float32, [None, 2])
    # Y_fake = tf.placeholder(tf.float32, [None, 2])

    # G
    z = tf.placeholder(tf.float32, [None, 10])  # noise
    tf.summary.histogram('Noise', z)
    G = generator(z, batch_size)
    tf.summary.image('generated image', G, 3)

    # D
    # Input = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y, Ylogits = discriminator(X, reuse=False)
    Y_, Ylogits_ = discriminator(G, reuse=True)

    with tf.name_scope('Prediction'):
        tf.summary.histogram('Raw', Y)
        tf.summary.histogram('Generated', Y_)

    with tf.name_scope('D_loss'):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ylogits, labels=tf.ones_like(Y)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ylogits_, labels=tf.zeros_like(Y_)))
        d_loss = d_loss_real + d_loss_fake
        tf.summary.scalar('d_loss_real', d_loss_real)
        tf.summary.scalar('d_loss_fake', d_loss_fake)
        tf.summary.scalar('d_loss', d_loss)

    with tf.name_scope('G_loss'):
        g_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ylogits_, labels=tf.ones_like(Y_)))
        g_loss = g_loss_d
        tf.summary.scalar('g_loss_d', g_loss_d)
        tf.summary.scalar('g_loss', g_loss)

    tvar = tf.trainable_variables()
    dvar = [var for var in tvar if 'discriminator' in var.name]
    gvar = [var for var in tvar if 'generator' in var.name]

    with tf.name_scope('train'):
        d_train_step = tf.train.AdamOptimizer(1e-5).minimize(d_loss, var_list=dvar)
        g_train_step = tf.train.AdamOptimizer(1e-5).minimize(g_loss, var_list=gvar)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tmp/mnist/3')
    writer.add_graph(sess.graph)

    for i in range(2500):
        batch_X, _ = mnist.train.next_batch(batch_size)
        batch_X = np.reshape(batch_X, [-1, 28, 28, 1])
        batch_noise = np.random.rand(batch_size, 10)

        # train G
        g_loss_print, batch_G, _ = sess.run([g_loss, G, g_train_step],
                                            feed_dict={X: batch_X, z: batch_noise},)
        # # train G twice
        # g_loss_print, batch_G, _ = sess.run([g_loss, G, g_train_step],
        #                                     feed_dict={X: batch_X, z: batch_noise})
        # train D
        if i % 1000 == 0:
        	d_loss_print, _ = sess.run([d_loss, d_train_step],
                                   feed_dict={X: batch_X, z: batch_noise})

        if i % 10 == 0:
            s = sess.run(merged_summary, feed_dict={X: batch_X, z: batch_noise})
            writer.add_summary(s, i)
            try:
                print('epoch:%d g_loss:%f d_loss:%f' % (i, g_loss_print, d_loss_print))
            except:
                pass

        # if i % 100 == 0:
        #     try:
        #         cv2.imwrite(os.path.join('..', 'figure', '1.png'), np.repeat(batch_G[0], 3, axis=2))
        #     except:
        #         pass

if __name__ == '__main__':
    train()
