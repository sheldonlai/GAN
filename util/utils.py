import os
import subprocess
from shutil import copyfile

import imageio as imageio
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from math import sqrt

from PIL import Image
from math import ceil


def analyze_sound(filename):
    sig, samplerate = sf.read(filename)
    plt.plot(sig)
    plt.show()


def get_sound_data(strategy=None):
    asset_path = "./assets"
    file_names = [os.path.join(asset_path, f) for f in os.listdir(asset_path)
                  if os.path.isfile(os.path.join(asset_path, f)) and f[-4:] == '.wav']
    if len(file_names) == 0:
        fetch_sample_files()

    a = []
    max_len = 0
    samplerate = None
    if strategy == "SQR":
        for filename in file_names:
            sig, samplerate = sf.read(filename)
            a.append(sig)
            if max_len < len(sig):
                max_len = len(sig)

        max_len = pow(round(sqrt(len(a[0]))), 2)
        for i, el in enumerate(a):
            if max_len > len(el):
                a[i] = np.append(el, np.zeros(max_len - len(el)))
            else:
                a[i] = el[:max_len]
        return a, samplerate
    else:
        for filename in file_names:
            sig, samplerate = sf.read(filename)
            a.append(sig)
        partition = 36000
        # create samples of 36000 length
        new_a = []
        for el in a:
            for i in range(len(el) // partition - 1):
                new_a.append(el[partition * i:partition * (i + 1)])
        return new_a, samplerate


def convert_sound_array_to_matrix(array, strategy=None):
    if strategy == "SQR":
        # Construct a matrix such that
        y_dim = round(sqrt(len(array[1])))
        x_dim = len(array[1]) // y_dim
        size = len(array)
        for i in range(size):
            array[i] = array[i].reshape([x_dim, y_dim, -1])
        return array, x_dim, y_dim, size
    elif strategy == "11khz":
        # supports up to 11khz
        y_dim = 256
        x_dim = 858
    else:
        return np.array(array).reshape(len(array), len(array[0]), 1, 1), len(array[0]), 1, len(array)


def write_image_matrix(array, name, strategy=None):
    dst_path = './image_out/'
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    imageio.imwrite(os.path.join(dst_path, name + '.jpg'), array)


def get_cifar_data():
    return


def write_sound_matrix_to_wav(matrix, rate, name, strategy=None, dir='./out/'):
    if strategy == "SQR":
        sf.write(os.path.join(dir, name + ".wav"), np.array(matrix).flatten(), rate)
    else:
        print("No strategy, not writing file.")


def fetch_sample_files():
    # copy the files to assets
    path = "C:\\Users\\SHELDON\\Music\\fma_small\\029"

    file_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
                  for f in filenames if os.path.splitext(f)[1] == '.mp3']
    dst_path = "./assets"
    file_names = file_names[0:min(4000, len(file_names))]
    for filename in file_names:
        subprocess.call(
            ["./util/ffmpeg.exe", "-i", filename, "-vn", "-acodec", "pcm_s16le", "-ac", "1",
             "-ar", "8000", "-f", "wav", os.path.join(dst_path, os.path.basename(filename)[:-4] + '.wav')])


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_cifar10_dict():
    data = {}
    path = './cifar/cifar-10-batches-py'
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


def cifar10_dict_to_matrix(label, data):
    return np.swapaxes(np.array(data[str(label)]).reshape([len(data[str(label)]), 3, 32, 32]), 1, 3), 32, 32, 3


def cifar10_batch_to_matrix(data):
    return np.swapaxes(data[b'data'].reshape([len(data[b'data']), 3, 32, 32]), 1, 3), 32, 32, 3


def combine_image_arrays(data, img_dim):
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
    return matrix.astype(np.uint8)


def get_cifar10_batch():
    data = []
    path = './cifar/cifar-10-batches-py'
    file_names = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isfile(os.path.join(path, f)) and "data_batch" in f]
    for file_path in file_names:
        data.append(unpickle(file_path))
    return data, unpickle(os.path.join(path, 'test_batch'))


def fetch_image_files():
    # copy the files to cat_images
    path = "C:\\Users\\SHELDON\\Pictures\\cats"

    size = 256

    file_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
                  for f in filenames if f[-4:] == '.jpg']
    dst_path = "./train_images"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    file_names = file_names[0:min(4000, len(file_names))]
    for filename in file_names:
        if os.path.isfile(filename + ".cat"):

            with open(filename + ".cat") as cat_file:
                line = cat_file.readline().rstrip()
            array = np.array(line.split(" ")[1:]).astype(np.int).reshape([-1, 2])
            print(filename)
            max_x, max_y = np.max(array, axis=0)
            print(max_x, max_y)
            min_x, min_y = np.min(array, axis=0)
            print(min_x, min_y)

            img = Image.open(filename)
            left, upper, right, lower = img.getbbox()
            if lower < size or right < size:
                continue
            img2 = crop_image(img, max_x, max_y, min_x, min_y)
            img2.save(os.path.join(dst_path, os.path.basename(filename)), "JPEG")
        else:
            copyfile(filename, os.path.join(dst_path, os.path.basename(filename)))


def crop_image(img, max_x, max_y, min_x, min_y, size=256):
    """
    :param size:
    :param img: PIL image
    :param max_x:
    :param max_y:
    :param min_x:
    :param min_y:
    :return: cropped and rescaled image of size x size
    """
    max_dif = max(max_x - min_x, max_y - min_y)
    padding = ceil(max_dif / 2)
    left, upper, right, lower = img.getbbox()
    # assert left <= min_x, "left: %d, min_x: %d" % (left, min_x)
    # assert right >= max_x, "right: %d, max_x: %d" % (right, max_x)
    # assert upper <= max_y
    # assert lower >= min_y

    # shrink image if the points difference is larger than size
    # if max_dif > size:
    #     factor = size/max_dif
    #     img.thumbnail((lower*factor, right*factor))
    #     max_y *= factor
    #     min_y *= factor
    #     max_x *= factor
    #     min_x *= factor

    # make height and width equal
    if max_x - min_x > max_y - min_y:
        dif = max_x - min_x - (max_y - min_y)
        max_y += dif // 2
        min_y -= ceil(dif / 2)

    elif max_x - min_x < max_y - min_y:
        dif = abs(max_x - min_x - (max_y - min_y))
        max_x += dif // 2
        min_x -= ceil(dif / 2)

    # make sure padding + max or min is within bounds
    # if not decrease padding
    if left > min_x - padding:
        padding = abs(left - min_x)
    if right < max_x + padding:
        padding = right - max_x
    if upper > max_y + padding:
        padding = upper - max_x
    if lower < min_y - padding:
        padding = abs(lower - max_x)

    img2 = img.crop((min_x - padding, min_y - padding, max_x + padding, max_y + padding))
    img2.thumbnail((size, size))
    return img2


def get_cifar10_gan_data(label=0):
    data, labels = get_cifar10_dict()

    return cifar10_dict_to_matrix(label, data)


def read_image_data():
    path = "./train_images"

    files = [os.path.join(path, fn) for fn in os.listdir(path) if fn[-4:] == '.jpg']
    data = []
    for fn in files:
        d = imageio.imread(fn)
        if d.shape == (256, 256, 3):
            data.append(d)
    print("have %d samples" % len(data))
    return data, 256, 256, 3


def generate_video_from_images(path):
    from skvideo.io import vwrite
    import cv2
    import os

    image_folder = path
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith("output")]

    data = []
    for image in images:
        data.append(cv2.imread(os.path.join(image_folder, image))[:, :2048])

    vwrite(os.path.join('./sample', video_name), np.array(data))
    # subprocess.call(
    #     ["./util/ffmpeg", '-r', '10', '-i', './image_out/output_%d.jpg', '-vcodec', 'mpeg4', '-y', 'movie.mp4'])
