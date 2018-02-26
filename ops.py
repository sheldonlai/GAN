import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm


# modified code from https://github.com/openai/improved-gan/blob/master/imagenet/ops.py
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


# modified code from https://github.com/openai/improved-gan/blob/master/imagenet/ops.py
def conv_transpose(input_, output_shape,
                   k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                   name="deconv2d", with_w=False,
                   init_bias=0):
    with tf.variable_scope(name):
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


# modified code from https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
def residual_block(input_layer, output_channel, first_block=False, block_num=None):
    """
    Defines a residual block in ResNet
    :param block_num: number of the block (used for naming scope)
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    """

    block_num = "_" + x_str(block_num)
    input_channel = input_layer.get_shape()[-1]
    with tf.variable_scope("block" + block_num):
        # When it's time to "shrink" the image size, we use stride =
        # output channel can be >= input channel since the first block would go from 3 to 64
        if input_channel * 2 <= output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block' + block_num):
            if first_block:
                conv1 = conv2d(input_layer, output_channel, name='conv2d_1', d_h=stride, d_w=stride, k_w=7, k_h=7)
            else:
                conv1 = tf.nn.leaky_relu(
                    batch_norm(conv2d(input_layer, output_channel, name='conv2d_1', d_h=stride, d_w=stride, k_h=3, k_w=3),
                               scope="bn1"))

        with tf.variable_scope('conv2_in_block' + block_num):
            conv2 = tf.nn.leaky_relu(batch_norm(conv2d(
                conv1, output_channel, name='conv2d_2', d_h=1, d_w=1, k_h=3, k_w=3), scope="bn2"))

        if first_block:
            return conv2

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            residual_orig_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                                 input_channel // 2]])
        else:
            residual_orig_input = input_layer

        output = conv2 + residual_orig_input
        return output


def x_str(s):
    if s is None:
        return ''
    return str(s)


def conv_transpose_layer(input_layer, output_channel, last_block=False, block_num=None):
    # custom implementation of a residual_block that does transpose convolution
    block_num = x_str(block_num)

    bn, dim_x, dim_y, input_channel = input_layer.get_shape().as_list()

    if input_channel // 2 == output_channel:
        stride = 2
        output_dim = [bn, dim_x * 2, dim_y * 2, int(output_channel)]
    elif input_channel == output_channel:
        stride = 1
        output_dim = [bn, dim_x, dim_y, int(output_channel)]
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    with tf.variable_scope('conv1_in_block_' + block_num):
        if last_block:
            conv1 = conv_transpose(input_layer, output_dim, name='conv2d_1', d_h=stride, d_w=stride, k_w=7, k_h=7)
        else:
            conv1 = tf.nn.relu(
                batch_norm(
                    conv_transpose(input_layer, output_dim, name='conv2d_1', d_h=stride, d_w=stride),
                    scope="bn1"))
    return conv1


def residual_transpose_block(input_layer, output_channel, last_block=False, block_num=None):
    # custom implementation of a residual_block that does transpose convolution
    block_num = x_str(block_num)

    bn, dim_x, dim_y, input_channel = input_layer.get_shape().as_list()

    if input_channel // 2 >= output_channel:
        decrease_dim = True
        stride = 2
        output_dim = [bn, dim_x * 2, dim_y * 2, int(output_channel)]
    elif input_channel == output_channel:
        decrease_dim = False
        stride = 1
        output_dim = [bn, dim_x, dim_y, int(output_channel)]
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    with tf.variable_scope('conv1_in_block_' + block_num):
        if last_block:
            conv1 = conv_transpose(input_layer, output_dim, name='conv2d_1', d_h=stride, d_w=stride, k_w=3, k_h=3)
        else:
            conv1 = tf.nn.relu(
                batch_norm(
                    conv_transpose(input_layer, output_dim, name='conv2d_1', d_h=stride, d_w=stride, k_w=3, k_h=3),
                    scope="bn1"))
    with tf.variable_scope('conv2_in_block' + block_num):
        conv2 = tf.nn.relu(
            batch_norm(conv_transpose(conv1, output_dim, name='conv2d_2', d_h=1, d_w=1, k_w=3, k_h=3),
                       scope="bn2"))

    # conversion from depth (e.g. 64 to 3 channels) then just return conv2
    if last_block:
        return conv2

    if decrease_dim:
        residual_orig_input = reverse_average_pool(input_layer, output_channel)
    else:
        residual_orig_input = input_layer

    # print("conv1: %s, conv2 : %s, residual_orig_input: %s" %
    #       (conv1.get_shape(), conv2.get_shape(), residual_orig_input.get_shape()))
    output = conv2 + residual_orig_input
    return output


def reverse_average_pool(input_layer, output_dim):
    """
    Custom implementation to perform something like a reverse average pool
    :param input_layer: with dimension (batch num, x, y, d)
    :return: a tensor with dimension (batch num, x * 2 , y * 2, d // 2)
    """

    bn, x, y, d = input_layer.get_shape()
    if output_dim == d // 2:
        pooled_input = tf.reduce_mean(tf.reshape(input_layer, [bn, x, y, d//2, 2]), 4)
        output = tf.image.resize_images(pooled_input, [x * 2, y * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return output
    else:
        raise ValueError("d: [%d] is not divisible by 2" % d)


def dense(x, outputFeatures, name="dense"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        return tf.layers.dense(x, outputFeatures)
