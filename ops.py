import tensorflow as tf
import numpy as np


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv


def conv1d(input_, output_dim, k_w=5, d_w=2, stddev=0.02, name="conv1d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(input_, w, stride=d_w, padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def conv_transpose(input_, output_shape,
                   k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                   name="deconv2d", with_w=False,
                   init_bias=0):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(init_bias))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def conv1d_transpose(input_, output_shape, k_w=5, d_w=2, stddev=0.02, name="deconv1d", with_w=False, init_bias=0):
    """
    Custom conv1d_transpose function, have not been tested
    :param input_: input 3D tensor [batch_size, dim_x, depth]
    :param output_shape: [batch_size, dim_x, depth]
    :param k_w: kernel width
    :param d_w: stride width
    :param stddev: standard deviation for weights initialization
    :param name: scope name
    :param with_w: True to return weights and biases
    :param init_bias: init_bias for biases
    :return: deconv or deconv, w, biases
    """
    with tf.variable_scope(name):
        new_input = tf.expand_dims(input_, 1)
        # print(new_input.get_shape())
        w = tf.get_variable('w', [1, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        # define output shape for conv2d
        temp_output_shape = [output_shape[0], 1, output_shape[1], output_shape[-1]]

        deconv = tf.nn.conv2d_transpose(new_input, w, output_shape=temp_output_shape, strides=[1, 1, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(init_bias))

        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        deconv = tf.squeeze(deconv, axis=1)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def dense(x, outputFeatures, name="dense"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.layers.dense(x, outputFeatures)
