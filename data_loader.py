import math

import tensorflow as tf
import numpy as np

from util.cat_data_util import fetch_cat_image_files, read_cat_image_data
from util.utils import get_training_data_for_label, get_cifar10_gan_data_patch


class CatDataLoader(object):

    def __init__(self, dim=128):
        self.dim = dim
        fetch_cat_image_files(dim)

    def get_data(self):
        m_data, x_dim, y_dim, channels = read_cat_image_data(self.dim)
        print("loaded %d images" % len(m_data))
        return m_data, x_dim, y_dim, channels


class CifarLoader(object):

    def __init__(self, data_batch=0, scale=1.0, data_sep_type="mixed"):
        """
        :param data_batch: data_batch is in range [0,9]
        :param scale: float in range [0 ,1.0]
        """
        if data_batch > 9 or data_batch < 0:
            raise Exception("data_batch out of range")
        elif scale > 1.0 or scale < 0:
            raise Exception("scale out of range")
        self.data_sep_type = data_sep_type
        self.data_batch = data_batch
        self.scale = scale

    def get_data(self):
        if self.data_sep_type == "class":
            m_data, x_dim, y_dim, channels = get_training_data_for_label(self.data_batch)
        else:
            batch_num = self.data_batch // 2
            mod = self.data_batch % 2
            m_data, x_dim, y_dim, channels = get_cifar10_gan_data_patch(batch_num)
            length = len(m_data)
            if mod == 0:
                m_data = m_data[:length//2]
            else:
                m_data = m_data[length//2:]

        if self.scale != 1.0:
            x_dim = int(math.floor(x_dim * self.scale))
            y_dim = int(math.floor(y_dim * self.scale))
            with tf.Session() as sess:
                m_data = sess.run(tf.image.resize_images(m_data, np.array([x_dim, y_dim])))

        return m_data, x_dim, y_dim, channels
