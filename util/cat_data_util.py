import os
from shutil import copyfile

import imageio
import numpy as np
from PIL import Image

from util.download import maybe_download_and_extract
from math import ceil

url1 = 'https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip'
url2 = 'https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip'
download_dir = './data/cat'


def maybe_download_and_extract_cat_data():
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    maybe_download_and_extract(url1, download_dir)
    maybe_download_and_extract(url2, download_dir)


def read_cat_image_data(dim=256, check_black_borders=True):
    path = os.path.join(download_dir, "train_images")

    files = [os.path.join(path, fn) for fn in os.listdir(path) if fn[-4:] == '.jpg']
    data = []
    files_not_used = []
    for fn in files:
        try:
            d = imageio.imread(fn)
            if check_black_borders:
                top_row = d[0, :, :]
                bot_row = d[-1, :, :]
                left_col = d[:, -1, :]
                right_col = d[:, -1, :]
                if np.array_equal(np.zeros_like(right_col), right_col) or np.array_equal(np.zeros_like(left_col),
                                                                             left_col) or np.array_equal(
                        np.zeros_like(bot_row), bot_row) or np.array_equal(np.zeros_like(top_row), top_row):
                    files_not_used.append(fn)
                    continue

            if d.shape == (dim, dim, 3):
                data.append(d)
        except Exception as e:
            print("cannot read file %s" % fn)
    print("have %d samples" % len(data))
    print("%d files are not used" % len(files_not_used))
    return np.array(data), dim, dim, 3


def fetch_cat_image_files(size=256):
    # copy the files to cat_images
    dst_path = os.path.join(download_dir, "train_images")
    if os.path.exists(dst_path):
        print("train_images directory already exists")
        return

    maybe_download_and_extract_cat_data()
    path = download_dir
    file_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
                  for f in filenames if f[-4:] == '.jpg']

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
            img2 = crop_image(img, max_x, max_y, min_x, min_y, size=size)
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
    #     max_y = ceil(max_y * factor)
    #     min_y = ceil(min_y * factor)
    #     max_x = ceil(max_x * factor)
    #     min_x = ceil(min_x * factor)
    #     left, upper, right, lower = img.getbbox()
    init_y_bias = 18

    # make height and width equal
    if max_x - min_x > max_y - min_y:
        dif = max_x - min_x - (max_y - min_y)
        max_y += dif // 2
        min_y -= ceil(dif / 2)

    elif max_x - min_x < max_y - min_y:
        dif = abs(max_x - min_x - (max_y - min_y))
        max_x += dif // 2
        min_x -= ceil(dif / 2)

    assert abs(max_x - min_x) == abs(max_y - min_y), 'dif x %f doesn\'t match dif y %f' % \
                                                     (abs(max_x - min_x), abs(max_y - min_y))

    padding = ceil(max(0, size - abs(max_x - min_x)) / 2)
    horizontal_shift = 0
    vertical_shift = 0

    def left_border():
        return min_x - padding + horizontal_shift

    def right_border():
        return max_x + padding + horizontal_shift

    def top_border():
        return min_y - padding + vertical_shift + init_y_bias

    def bot_border():
        return max_y + padding + vertical_shift + init_y_bias

    while True:
        if left > left_border() - padding and horizontal_shift >= 0:
            horizontal_shift += 1
        if right < right_border() and horizontal_shift <= 0:
            horizontal_shift -= 1
        if upper > top_border() and vertical_shift >= 0:
            vertical_shift += 1
        if lower < bot_border() - padding and vertical_shift <= 0:
            vertical_shift -= 1
        else:
            break

    # make sure padding + max or min is within bounds
    # if not decrease padding
    if left > left_border():
        padding = abs(left_border() - left)
    if right < right_border():
        padding = abs(right_border() - right)
    if upper > top_border():
        padding = abs(top_border() - upper)
    if lower < bot_border():
        padding = abs(bot_border() - lower)

    img2 = img.crop((left_border(), top_border(), right_border(), bot_border()))
    img2.thumbnail((size, size))
    return img2
