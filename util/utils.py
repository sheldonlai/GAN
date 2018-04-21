import os
import subprocess
from shutil import copyfile

import imageio as imageio
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import re
from skvideo.io import FFmpegWriter

from util.downloader import maybe_download_and_extract


def write_image_matrix(array, name, dst_folder=None):
    if dst_folder is None:
        dst_folder = './image_out/'
    else:
        dst_folder = os.path.join(os.getcwd(), dst_folder)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    imageio.imwrite(os.path.join(dst_folder, name + '.png'), array)


def get_cifar_data():
    return


def write_sound_matrix_to_wav(matrix, rate, name, strategy=None, dir='./out/'):
    if strategy == "SQR":
        sf.write(os.path.join(dir, name + ".wav"), np.array(matrix).flatten(), rate)
    else:
        print("No strategy, not writing file.")


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def _maybe_download_cifar10_data():
    path = './data/cifar10'

    if not os.path.exists(path):
        os.makedirs(path)

    maybe_download_and_extract(path)


def get_cifar10_dict():
    data = {}
    path = './data/cifar10/cifar-10-batches-py'

    _maybe_download_cifar10_data()

    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "data_batch" in f]

    for file_path in file_names:
        temp = unpickle(file_path)
        for i in range(len(temp[b'data'])):
            if str(temp[b'labels'][i]) in data:
                data[str(temp[b'labels'][i])].append(temp[b'data'][i])
            else:
                data[str(temp[b'labels'][i])] = [temp[b'data'][i]]

    return data, unpickle(os.path.join(path, 'test_batch'))


def get_training_data_for_label(label):
    d, t = get_cifar10_dict()
    return cifar10_dict_to_matrix(label, d)


def cifar10_dict_to_matrix(label, data):
    return np.swapaxes(np.swapaxes(np.array(data[str(label)]).reshape([len(data[str(label)]), 3, 32, 32]), 1, 3), 1,
                       2), 32, 32, 3


def cifar10_batch_to_matrix(data):
    return np.swapaxes(np.swapaxes(data[b'data'].reshape([len(data[b'data']), 3, 32, 32]), 1, 3), 1, 2), 32, 32, 3


def combine_image_arrays(data, img_dim, convert_tanh_to_unit=True):
    assert len(img_dim) == 2, "img_dim should be length 2 array"
    batch_size, x_dim, y_dim, channels = data.shape

    h, w = [round(e) for e in img_dim]
    assert batch_size == h * w, "batch_size should equal img_dim[0] * img_dim[1]"

    matrix = np.array([])
    for i in range(h):
        row_array = [np.reshape(data[i * w + j], [x_dim, y_dim, channels]) for j in range(w)]
        row = np.concatenate(row_array, axis=1)
        if len(matrix) == 0:
            matrix = row
        else:
            matrix = np.vstack([matrix, row])
    if convert_tanh_to_unit:
        matrix = (matrix + 1) * 127.5
    return matrix.astype(np.uint8)


def get_cifar10_batch():
    data = []
    path = './data/cifar10/cifar-10-batches-py'
    _maybe_download_cifar10_data()
    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "data_batch" in f]
    for file_path in file_names:
        data.append(unpickle(file_path))
    return data, unpickle(os.path.join(path, 'test_batch'))


def get_cifar10_gan_data(label=0):
    data, labels = get_cifar10_dict()

    return cifar10_dict_to_matrix(label, data)


def get_cifar10_gan_data_patch(num=0):
    path = os.path.join(os.getcwd(), './data/cifar10/cifar-10-batches-py')
    _maybe_download_cifar10_data()
    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "data_batch" in f]
    data = unpickle(file_names[num])

    return cifar10_batch_to_matrix(data)


def generate_video_from_images(path):
    from skvideo.io import vwrite
    import cv2
    import os

    image_folder = path
    video_name = 'video_3.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith("output")]
    images = sorted(images, key=lambda x: (int(re.sub('\D', '', x)), x))
    writer = FFmpegWriter(os.path.join('./sample', video_name),
                          inputdict={'-r': '3'}, outputdict={'-r': '3'})

    for image in images:
        writer.writeFrame(cv2.imread(os.path.join(image_folder, image))[:, :2048])
    writer.close()
    # subprocess.call(
    #     ["./util/ffmpeg", '-r', '10', '-i', './image_out/output_%d.jpg', '-vcodec', 'mpeg4', '-y', 'movie.mp4'])
