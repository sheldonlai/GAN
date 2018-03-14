import time
from math import ceil

import gc
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

from ops import conv2d, dense, conv_transpose, conv1d, conv1d_transpose, residual_block, residual_transpose_block, \
    conv_transpose_layer
from util.utils import *

tf.logging.set_verbosity(tf.logging.DEBUG)


def train(train=True, output_name='output'):
    def discriminator(data, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if False and x_dim >= 128 and y_dim > 128:
            # ResNet
            layers = []
            for i in range(res_net_layers):
                if i == 0:
                    layers.append(residual_block(data, df_dim, True, block_num=i))
                elif i % res_net_bc == 0:
                    layers.append(residual_block(layers[i - 1], layers[i - 1].get_shape()[-1] * 2,
                                                 False, block_num=i, avg_pool_add=False))
                else:
                    layers.append(residual_block(layers[i - 1], layers[i - 1].get_shape()[-1], False, block_num=i))

            dense_output_layer = dense(layers[-1], 1, name="dense_" + str(res_net_layers))
            return tf.nn.sigmoid(dense_output_layer), dense_output_layer
        elif y_dim > 1:
            prev_layer = tf.nn.leaky_relu(batch_norm(conv2d(data, df_dim, name='d_h0_conv'), scope="bn_0"))
            for i in range(1,num_layers - 2):
                prev_layer = tf.nn.leaky_relu(batch_norm(conv2d(prev_layer, df_dim * pow(2,i),
                                                                name='d_h'+str(i)+'_conv'), scope="bn_"+str(i)))
            prev_layer = tf.nn.relu(batch_norm(conv2d(prev_layer, df_dim * 16, name='d_h4_conv'), scope="bn_4"))
            dense_layer = dense(tf.reshape(prev_layer, (batch_size, -1)), 1)
            # h0 = tf.nn.leaky_relu(batch_norm(conv2d(data, df_dim, name='d_h0_conv'), scope="bn_0"))
            # h1 = tf.nn.leaky_relu(batch_norm(conv2d(h0, df_dim * 2, name='d_h1_conv'), scope="bn_1"))
            # h2 = tf.nn.leaky_relu(batch_norm(conv2d(h1, df_dim * 4, name='d_h2_conv'), scope="bn_2"))
            # h3 = tf.nn.leaky_relu(batch_norm(conv2d(h2, df_dim * 8, name='d_h3_conv'), scope="bn_3"))
            # h4 = tf.nn.relu(batch_norm(conv2d(h3, df_dim * 16, name='d_h4_conv'), scope="bn_4"))
            # h5 = dense(h4, 1)
            return tf.nn.sigmoid(dense_layer), dense_layer
        else:
            h0 = tf.nn.relu(batch_norm(conv1d(data, df_dim, name='d_h0_conv'), scope="bn_0"))
            h1 = tf.nn.relu(batch_norm(conv1d(h0, df_dim * 2, k_w=8, name='d_h1_conv'), scope="bn_1"))
            h2 = tf.nn.relu(batch_norm(conv1d(h1, df_dim * 4, k_w=16, name='d_h2_conv'), scope="bn_2"))
            h3 = tf.nn.relu(batch_norm(conv1d(h2, df_dim * 8, k_w=32, name='d_h3_conv'), scope="bn_3"))
            h4 = dense(h3, 1)
            return tf.nn.sigmoid(h4), h4

    def get_dim2d(layer):
        return [batch_size, ceil(x_dim / pow(2, layer)), ceil(y_dim / pow(2, layer)), gf_dim * pow(2, layer - 1)]

    def get_dim1d(layer):
        return [batch_size, get_x_dim1d(layer), gf_dim * pow(2, layer - 1)]

    def get_x_dim1d(layer):
        return ceil(x_dim / pow(2, layer))

    def generator(z):

        if x_dim >= 128 and y_dim > 128 and res_net:
            _, dim0, dim1, gf_dim_last = get_dim2d(res_net_layers // res_net_bc)
            z2 = dense(z, dim0 * dim1 * gf_dim_last)
            # ResNet transpose
            layers = []
            for i in range(res_net_layers):
                if i == 0:
                    layers.append(residual_transpose_block(
                        tf.reshape(z2, [batch_size, dim0, dim1, gf_dim_last]), gf_dim_last, True, block_num=i))
                elif i % res_net_bc == 0:
                    layers.append(
                        residual_transpose_block(layers[i - 1], layers[i - 1].get_shape()[-1] // 2,
                                                 False, block_num=i, avg_pool_add=False))
                else:
                    layers.append(
                        residual_transpose_block(layers[i - 1], layers[i - 1].get_shape()[-1], False, block_num=i))

            last_layer = residual_transpose_block(layers[-1], 3, last_block=True, block_num=res_net_layers)

            return (tf.nn.tanh(last_layer) + 1) * 255 / 2
        elif y_dim > 1:
            # _, dim0, dim1, gf_dim_last = get_dim2d(5)
            # z2 = dense(z, dim0 * dim1 * gf_dim_last)
            # h0 = tf.nn.relu(batch_norm(tf.reshape(z2, [-1, dim0, dim1, gf_dim_last])))
            # h05 = tf.nn.relu(batch_norm(conv_transpose(h0, get_dim2d(4), name="g_h1")))
            # h1 = tf.nn.relu(batch_norm(conv_transpose(h05, get_dim2d(3), name="g_h05")))
            # h2 = tf.nn.relu(batch_norm(conv_transpose(h1, get_dim2d(2), name="g_h2")))
            # h3 = tf.nn.relu(batch_norm(conv_transpose(h2, get_dim2d(1), name="g_h3")))
            # h4 = conv_transpose(h3, [batch_size, x_dim, y_dim, channels], name="g_h4")
            # return (tf.nn.tanh(h4) + 1) * 255 / 2
            _, dim0, dim1, gf_dim_last = get_dim2d(num_layers-1)
            z2 = dense(z, dim0 * dim1 * gf_dim_last)
            prev_layer = tf.nn.relu(batch_norm(tf.reshape(z2, [-1, dim0, dim1, gf_dim_last])))
            for i in range(1,num_layers - 1):
                prev_layer = tf.nn.relu(batch_norm(conv_transpose(prev_layer, get_dim2d(num_layers-i-1), name="g_h" + str(i))))
            hlast = conv_transpose(prev_layer, [batch_size, x_dim, y_dim, channels], name="g_h" + str(num_layers - 1))
            return (tf.nn.tanh(hlast) + 1) * 255 / 2
        else:
            z2 = dense(z, get_x_dim1d(4) * gf_dim * 8)
            h0 = tf.nn.relu(batch_norm(tf.reshape(z2, get_dim1d(4))))
            h1 = tf.nn.relu(batch_norm(conv1d_transpose(h0, get_dim1d(3), k_w=8, name="g_h1")))
            h2 = tf.nn.relu(batch_norm(conv1d_transpose(h1, get_dim1d(2), k_w=16, name="g_h2")))
            h3 = tf.nn.relu(batch_norm(conv1d_transpose(h2, get_dim1d(1), k_w=32, name="g_h3")))
            h4 = conv1d_transpose(h3, [batch_size, x_dim, channels], name="g_h4")
            return tf.nn.tanh(h4)

    def get_samples(batch_index):

        if (batch_index + 1) * batch_size <= len(m_data):
            res = m_data[batch_index * batch_size:(batch_index + 1) * batch_size]
        else:
            res = m_data[batch_index * batch_size:] + m_data[:len(m_data) - (batch_index * batch_size)]

        res = np.array(res).astype(np.float32)
        if y_dim == 1:
            return np.squeeze(res, axis=2)
        else:
            return res

    data, labels = get_cifar10_batch()
    m_data, x_dim, y_dim, channels = cifar10_batch_to_matrix(data[0])

    print(len(m_data))
    # resnet variables
    res_net = False
    res_net_layers = 8

    # number of block before the residual network change in dimension
    res_net_bc = 2

    # normal convolution variables
    num_layers = 6

    batch_size = 64
    if y_dim == 1:
        data_shape = [x_dim, channels]
    else:
        data_shape = [x_dim, y_dim, channels]
    z_dim = 100
    gf_dim = 64
    df_dim = 64
    learning_rate = 0.000002
    beta1 = 0.5

    with tf.Session() as sess:
        # build model
        images = tf.placeholder(tf.float32, [batch_size] + data_shape, name="real")
        zin = tf.placeholder(tf.float32, [None, z_dim], name="z")
        with tf.variable_scope("generator"):
            generated = generator(zin)
        with tf.variable_scope("discriminator"):
            d_prob, d_logit = discriminator(images)
            d_generated_prob, d_generated_logit = discriminator(generated, True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit, labels=tf.ones(d_logit.shape)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_generated_logit, labels=tf.zeros(d_generated_logit.shape)))

        gloss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_generated_logit, labels=tf.ones(d_generated_logit.shape)))
        dloss = d_loss_real + d_loss_fake

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        with tf.control_dependencies(d_update_ops):
            d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(dloss, var_list=d_vars)

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        with tf.control_dependencies(g_update_ops):
            g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(gloss, var_list=g_vars)

        display_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

        global_step = tf.Variable(1, trainable=False, name='global_step')
        increment_global_step_op = tf.assign(global_step, global_step + 1)

        saver = tf.train.Saver(max_to_keep=10)

        sess.run(tf.global_variables_initializer())

        start_time = time.time()

        if train:
            try:
                saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + "/training/"))
                counter = tf.train.global_step(sess, global_step)
                print("Recover session at step: %d." % counter)
            except:
                print("Unable to recover session starting anew.")

            print("starting")
            for epoch in range(1000):
                batch_idx = (len(m_data) // batch_size) - 2
                for idx in range(batch_idx):
                    batch_images = get_samples(idx)

                    batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

                    sess.run([d_optim], feed_dict={images: batch_images, zin: batch_z})

                    sess.run([g_optim, increment_global_step_op], feed_dict={zin: batch_z})

                    counter = tf.train.global_step(sess, global_step)

                    if counter % 500 == 0:
                        sdata = sess.run([generated], feed_dict={zin: display_z})
                        write_image_matrix(combine_image_arrays(sdata[0], [8, batch_size/8]),
                                           'output_' + str(counter))

                        errD_fake = d_loss_fake.eval({zin: display_z})
                        errD_real = d_loss_real.eval({images: batch_images})
                        errG = gloss.eval({zin: batch_z})
                        print("Step: %d, epoch: %d, time: %4.4f" % (counter, epoch, time.time() - start_time,))
                        print("errd: %4.4f errg: %4.4f" % (errD_real + errD_fake, errG))
                        start_time = time.time()
                    if counter % 1000 == 0:
                        saver.save(sess, os.getcwd() + "/training/train", global_step=global_step)

        else:
            saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + "/training/"))
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

            sdata = sess.run([generated], feed_dict={zin: batch_z})
            write_image_matrix(
                combine_image_arrays(sdata[0], [sqrt(batch_size), sqrt(batch_size)]),
                "nice")

        print("ending")
