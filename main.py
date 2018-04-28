import os
import math

import tensorflow as tf
import numpy as np

from data_loader import CatDataLoader, CifarLoader
from model import GANModel

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_num', 0, "batch integer in range [0,9]")

tf.app.flags.DEFINE_integer('batch_size', 512, "batch size")

tf.app.flags.DEFINE_integer('num_layers', 4, "number of layers (e.g. 4 will have 1 dense and 3 convolution layers)")

tf.app.flags.DEFINE_float('learning_rate', 0.0002, """learning_rate""")

tf.app.flags.DEFINE_integer('save', 100, "how many iterations between saves")

tf.app.flags.DEFINE_boolean('train', True, "Train network")

tf.app.flags.DEFINE_string('data_set', 'cifar', 'data set could be "cifar" or "cat"')

tf.app.flags.DEFINE_string('sub_dir', '', "subdirectory inside training and image_out folder")

tf.app.flags.DEFINE_string('generate_name', 'generated_sample', "name of generated file in GEN mode")
tf.app.flags.DEFINE_string('generate_type', 'random', "possible types 'random' 'arg_max' 'threshold'")
tf.app.flags.DEFINE_integer('generate_output_dim', 64, "name of generated file in GEN mode")


tf.app.flags.DEFINE_integer('filter_depth', 32, "filter depth")
tf.app.flags.DEFINE_integer('creativity', 100, "creativity")

chkpnt_dir = "training"
imageout_dir = "image_out"
if FLAGS.sub_dir != "":
    chkpnt_dir = os.path.join("training", FLAGS.sub_dir)
    imageout_dir = os.path.join("image_out", FLAGS.sub_dir)

if FLAGS.data_set is 'cifar':
    data_loader = CifarLoader(data_sep_type="class", data_batch=3)
else:
    data_loader = CatDataLoader()

model = GANModel(data_loader, batch_size=FLAGS.batch_size, num_layers=FLAGS.num_layers,
                 filter_depth=FLAGS.filter_depth, z_dim=FLAGS.creativity, train=FLAGS.train,
                 chkpnt_dir=chkpnt_dir, imageout_dir=imageout_dir, learning_rate=FLAGS.learning_rate)

if FLAGS.train:
    model.save_image(model.m_data[:model.batch_size], 'orig')
    print("starting model train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.recover_lastest_checkpoint(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 0
        while True:
            step, d_loss, g_loss = model.train(sess)
            if step % FLAGS.save == 0:
                model.print_results(step, sess, d_loss=d_loss, g_loss=g_loss)
                model.save_model(sess)

        # coord.request_stop()
        # coord.join(threads)
else:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.recover_lastest_checkpoint(sess)
        ok_images = []
        require_images = 121
        while len(ok_images) < require_images:
            images = model.generate_images(sess)
            prob = sess.run(tf.sigmoid(model.d_logit), feed_dict={model.images: images})
            if FLAGS.generate_type == 'arg_max':
                # write the image with the image with the highest prob
                max_index = np.argmax(prob)
                ok_images.append(images[max_index])
                print("max: %f%%" % prob[max_index] * 100)

            elif FLAGS.generate_type == 'threshold':
                for i in range(len(prob)):
                    if prob[i] > 0.3: # change the prob threshold
                        print("image looks real enough, added to the ok_images")
                        ok_images.append(images[i])
                        if len(ok_images) == require_images:
                            break
                    else:
                        print("image looks fake not added")
            else:
                ok_images = images

        dim = FLAGS.generate_output_dim
        # possible resizing
        if dim != ok_images.shape[1]:
            ok_images = sess.run(tf.image.resize_images(ok_images, np.array([dim, dim])))
        model.save_image(ok_images,
                         image_out_dim=int(math.sqrt(require_images)),
                         name=FLAGS.generate_name, directory="sample_result")
