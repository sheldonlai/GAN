import time

import tensorflow as tf

from data_loader import CatDataLoader, CifarLoader
from model import GANModel

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_num', 0, "batch integer in range [0,9]")

tf.app.flags.DEFINE_integer('batch_size', 512,"batch size")

tf.app.flags.DEFINE_integer('num_layers', 6, "number of layers (e.g. 4 will have 1 dense and 3 convolution layers)")

tf.app.flags.DEFINE_integer('save', 100, "how many iterations between saves")

tf.app.flags.DEFINE_boolean('train', True, "Train network")

tf.app.flags.DEFINE_string('data_set', 'cifar', 'data set could be "cifar" or "cat"')

if FLAGS.data_set is 'cifar':
    data_loader = CifarLoader(data_sep_type="class", data_batch=3)
else:
    data_loader = CatDataLoader()
model = GANModel(data_loader)
model.save_image(model.m_data[:model.batch_size], 'orig')
if FLAGS.train:
    print("starting self train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.recover_lastest_checkpoint(sess)
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        step = 0
        while step < 100000000000000:
            step = model.train(sess)
            if step % FLAGS.save == 0:
                model.print_results(step, sess)
                model.save_model(sess)

        # coord.request_stop()
        # coord.join(threads)
