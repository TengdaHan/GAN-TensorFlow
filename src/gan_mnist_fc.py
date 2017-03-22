# tf rd test - mnist GAN
import tensorflow as tf
import numpy as np
import cv2
import os


def conv(x, w, b, stride, name):
    with tf.variable_scope('conv'):
        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        return tf.nn.conv2d(x,
                           filter=w,
                           strides=[1, stride, stride, 1],
                           padding='SAME',
                           name=name) + b


def deconv(x, w, b, shape, stride, name):
    with tf.variable_scope('deconv'):
        tf.summary.histogram('weight', w)
        tf.summary.histogram('biases', b)
        return tf.nn.conv2d_transpose(x,
                                       filter=w,
                                       output_shape=shape,
                                       strides=[1, stride, stride, 1],
                                       padding='SAME',
                                       name=name) + b


def lrelu(x, alpha=0.2):
    with tf.variable_scope('leakyReLU'):
        return tf.maximum(x, alpha*x)


def read_data():
    # read mnist data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    return mnist


def discriminator(X, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        J = 784
        K = 128
        M = 1

        W1 = tf.Variable(tf.truncated_normal([J, K], stddev=0.1))
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        W2 = tf.Variable(tf.truncated_normal([K, M], stddev=0.1))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

        flat = tf.reshape(X, [-1, 784], name='flat')
        fc1 = lrelu(tf.matmul(flat, W1) + B1)
        logits = tf.matmul(fc1, W2) + B2
        return tf.nn.sigmoid(logits), logits


def generator(X, batch_size):
    with tf.variable_scope('generator'):
        K = 128
        M = 784

        W1 = tf.Variable(tf.truncated_normal([100, K], stddev=0.1))
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        W2 = tf.Variable(tf.truncated_normal([K, M], stddev=0.1))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

        fc1 = lrelu(tf.matmul(X, W1) + B1)
        fc2 = tf.matmul(fc1, W2) + B2
        reshape = tf.reshape(fc2, [batch_size, 28, 28, 1])

        return tf.nn.tanh(reshape)


def train(batch_size=100):
    mnist = read_data()

    with tf.variable_scope('placeholder'):
        # Raw image
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        z = tf.placeholder(tf.float32, [None, 100])  # noise

    tf.summary.image('raw image', X, 3)
    tf.summary.histogram('Noise', z)

    with tf.variable_scope('GAN'):
        G = generator(z, batch_size)

        D_real, D_real_logits = discriminator(X, reuse=False)
        D_fake, D_fake_logits = discriminator(G, reuse=True)
    tf.summary.image('generated image', G, 3)

    with tf.name_scope('Prediction'):
        tf.summary.histogram('Raw', D_real)
        tf.summary.histogram('Generated', D_fake)

    with tf.name_scope('D_loss'):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        # d_loss_real = -tf.reduce_mean(tf.log(D_real))
        # d_loss_fake = -tf.reduce_mean(tf.log(1. - D_fake))

        d_loss = d_loss_real + d_loss_fake
        tf.summary.scalar('d_loss_real', d_loss_real)
        tf.summary.scalar('d_loss_fake', d_loss_fake)
        tf.summary.scalar('d_loss', d_loss)

    with tf.name_scope('G_loss'):
        g_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
        # g_loss_d = -tf.reduce_mean(tf.log(D_fake))
        g_loss = g_loss_d
        tf.summary.scalar('g_loss_d', g_loss_d)
        tf.summary.scalar('g_loss', g_loss)

    tvar = tf.trainable_variables()
    dvar = [var for var in tvar if 'discriminator' in var.name]
    gvar = [var for var in tvar if 'generator' in var.name]

    with tf.name_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        # lr_d = tf.train.exponential_decay(1e-4, global_step, 100, 0.8, staircase=False)
        # lr_g = tf.train.exponential_decay(1e-3, global_step, 100, 0.4, staircase=False)

        d_train_step = tf.train.AdamOptimizer().minimize(d_loss, global_step=global_step, var_list=dvar)
        g_train_step = tf.train.AdamOptimizer().minimize(g_loss, global_step=global_step, var_list=gvar)

        # tf.summary.scalar('lr_d', lr_d)
        # tf.summary.scalar('lr_g', lr_g)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tmp/mnist/102')
    writer.add_graph(sess.graph)

    for i in range(100000):
        batch_X, _ = mnist.train.next_batch(batch_size)
        batch_X = np.reshape(batch_X, [-1, 28, 28, 1])
        batch_noise = np.random.uniform(-1., 1., [batch_size, 100])

        # train G
        # if i % 1000 == 0:
        g_loss_print, _ = sess.run([g_loss, g_train_step],
                                   feed_dict={X: batch_X, z: batch_noise},)

        # train D
        # if i % 1000 == 0:
        d_loss_print, _ = sess.run([d_loss, d_train_step],
                                   feed_dict={X: batch_X, z: batch_noise})

        if i % 25 == 0:
            s = sess.run(merged_summary, feed_dict={X: batch_X, z: batch_noise})
            writer.add_summary(s, i)
            try:
                print('epoch:%d g_loss:%f d_loss:%f' % (i, g_loss_print, d_loss_print))
            except:
                pass

        if i % 1000 == 0:
            try:
                batch_G = sess.run(G, feed_dict={X: batch_X, z: batch_noise})
                cv2.imwrite(os.path.join('..', 'figure', '2.png'), (np.repeat(batch_G[0], 3, axis=2)*255))
            except:
                pass

if __name__ == '__main__':
    train()
