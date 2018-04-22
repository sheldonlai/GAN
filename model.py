import tensorflow as tf
import time
from tensorflow.contrib.layers import batch_norm

from ops import conv2d, dense, conv_transpose
from util.utils import *
from math import ceil, floor, sqrt

tf.logging.set_verbosity(tf.logging.DEBUG)


def convert_tanh_float_to_uint(images):
    return (images + 1) * 127.5


class GANModel(object):
    def __init__(self, data_loader, num_layers=6, batch_size=128, learning_rate=0.00002,
                 z_dim=100, filter_depth=32, train=True,
                 chkpnt_dir="training", imageout_dir="image_out", scope="ml"):
        print("""
        Model:
        num_layers: %d
        batch_size: %d
        learning_rate: %f
        
        """ % (num_layers, batch_size, learning_rate))

        self.chkpnt_dir = os.path.join(os.getcwd(), chkpnt_dir)
        self.chkpnt_path = os.path.join(os.getcwd(), chkpnt_dir, 'ckpt')
        self.imageout_dir = imageout_dir
        self.batch_size = batch_size

        self.image_out_dim = int(floor(sqrt(batch_size)))

        self.z_dim = z_dim
        self.display_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        scope_prefix = scope + '/'
        with tf.variable_scope(scope):
            if train:
                m_data, x_dim, y_dim, channels = data_loader.get_data()
            else:
                m_data = []
                x_dim, y_dim, channels = data_loader.get_dim()

            data_shape = [x_dim, y_dim, channels]

            gf_dim = filter_depth
            df_dim = filter_depth
            self.learning_rate = learning_rate

            m_data = (m_data / 127.5) - 1
            self.m_data = m_data

            self.x_dim = x_dim
            self.y_dim = y_dim
            self.channels = channels

            beta1 = 0.5
            min_after_dequeue = batch_size
            num_threads = 2

            self.batch_len_in_epoch = (len(self.m_data) // self.batch_size) - 2

            capacity = batch_size * 8

            self.queue = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_after_dequeue,
                                               dtypes=tf.float32, shapes=data_shape)

            enqueue_op = self.queue.enqueue_many(self.m_data)

            qr = tf.train.QueueRunner(self.queue, [enqueue_op] * num_threads)
            tf.train.add_queue_runner(qr)

            self.last_time = time.time()

            def discriminator(data, reuse=False):
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                prev_layer = tf.nn.leaky_relu(batch_norm(conv2d(data, df_dim, name='d_h0_conv'), scope="bn_0"))
                for i in range(1, num_layers - 1):
                    prev_layer = tf.nn.leaky_relu(batch_norm(
                        conv2d(prev_layer, df_dim * pow(2, i), name='d_h' + str(i) + '_conv'), scope="bn_" + str(i)))
                dense_layer0 = tf.squeeze(dense(tf.reshape(prev_layer, (self.batch_size, -1)), 1))
                return dense_layer0

            def get_dim2d(layer):
                return [self.batch_size, ceil(x_dim / pow(2, layer)), ceil(y_dim / pow(2, layer)),
                        gf_dim * pow(2, layer - 1)]

            def generator(z):
                _, dim0, dim1, gf_dim_last = get_dim2d(num_layers - 1)
                z2 = dense(z, dim0 * dim1 * gf_dim_last)
                prev_layer = tf.nn.relu(batch_norm(tf.reshape(z2, [-1, dim0, dim1, gf_dim_last])))
                for i in range(1, num_layers - 1):
                    prev_layer = tf.nn.relu(
                        batch_norm(conv_transpose(prev_layer, get_dim2d(num_layers - i - 1), name="g_h" + str(i))))
                hlast = conv_transpose(prev_layer, [self.batch_size, x_dim, y_dim, channels],
                                       name="g_h" + str(num_layers - 1))
                return tf.nn.tanh(hlast)

            # build model
            if train:
                self.images = self.queue.dequeue_many(self.batch_size)
            else:
                self.images = tf.placeholder(tf.float32, [self.batch_size] + data_shape, name="real")

            self.zin = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="z")

            self.global_step = tf.Variable(1, trainable=False, name='global_step')
            self.increment_global_step_op = tf.assign(self.global_step, tf.add(self.global_step, tf.constant(1)))

            with tf.variable_scope("generator"):
                self.generated = generator(self.zin)
            with tf.variable_scope("discriminator"):
                self.d_logit = discriminator(self.images)
                d_generated_logit = discriminator(self.generated, True)

            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logit, labels=tf.ones(self.d_logit.shape)))
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_generated_logit,
                                                        labels=tf.zeros(d_generated_logit.shape)))

            self.gloss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_generated_logit,
                                                        labels=tf.ones(d_generated_logit.shape)))

            self.dloss = self.d_loss_real + self.d_loss_fake

            self.d_vars = [el for el in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                          scope=scope_prefix + 'discriminator')
                           if "bn" not in el.name]
            self.d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_prefix + 'discriminator')

            with tf.control_dependencies(self.d_update_ops):
                self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1)
                self.d_optim_op = self.d_optim.minimize(self.dloss, var_list=self.d_vars)

            self.g_vars = [el for el in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_prefix + 'generator')
                           if "bn" not in el.name]
            self.g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope_prefix + 'generator')

            with tf.control_dependencies(self.g_update_ops):
                self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1)
                self.g_optim_op = self.g_optim.minimize(self.gloss, var_list=self.g_vars)

            self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    def recover_lastest_checkpoint(self, sess):
        try:
            self.saver.restore(sess, tf.train.latest_checkpoint(self.chkpnt_dir))
            step = tf.train.global_step(sess, self.global_step)
            print("Recover session at step: %d." % step)
        except:
            print("Unable to recover session starting anew.")

    def get_batch(self, data, index):
        index = index % self.batch_len_in_epoch
        if (index + 1) * self.batch_size <= len(data):
            res = data[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            res = data[index * self.batch_size:] + \
                  data[:len(data) - (index * self.batch_size)]

        res = np.array(res).astype(np.float32)
        return res

    def get_training_batch(self, index):
        return self.get_batch(self.m_data, index)

    def get_variables(self):
        return self.d_vars + self.g_vars

    def calculate_loss_from_logits(self, logits, labels):
        """
        :param logits:
        :param labels:
        :return: an operation
        """
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    def train(self, sess):
        # images = self.get_training_batch(tf.train.global_step(sess, self.global_step) % self.batch_len_in_epoch)
        z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        sess.run([self.d_optim_op, self.g_optim_op, self.increment_global_step_op],
                 feed_dict={self.zin: z})
        return tf.train.global_step(sess, self.global_step)

    def save_model(self, sess):
        save_path = os.path.join(os.getcwd(), self.chkpnt_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.saver.save(sess, self.chkpnt_path, global_step=self.global_step)

    def save_image(self, images, name='pic'):
        if len(images) > self.batch_size:
            raise Exception('The batch size is smaller.')

        combined_images = combine_image_arrays(images[:self.image_out_dim ** 2],
                                               [self.image_out_dim, self.image_out_dim])
        write_image_matrix(combined_images, name, dst_folder=self.imageout_dir)

    def print_results(self, step, sess, start_time=None):
        if start_time is None:
            start_time = self.last_time
        sdata = sess.run(self.generated, feed_dict={self.zin: self.display_z})
        self.save_image(sdata[:self.image_out_dim ** 2], 'output_' + str(step))
        print('time eplased %4.4f, step: %d' % ((time.time() - start_time), step))
        self.last_time = time.time()
