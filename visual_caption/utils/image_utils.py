# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import skimage.color
import skimage.io
from scipy.misc import imresize


def load_image_raw(image_path):
    try:
        current_image = skimage.io.imread(image_path)
        # convert a gray to RGB image
        if len(current_image.shape) == 2 or current_image.shape[2] == 1:
            current_image = skimage.color.gray2rgb(current_image)  #
    except OSError as err:
        print(err)
        current_image = None
    return current_image


def load_images(image_files):
    raw_images = list()
    for idx, image_path in enumerate(image_files):
        image_rawdata = load_image_raw(image_path=image_path)
        if image_rawdata is not None:
            image_rawdata = imresize(image_rawdata, (299, 299, 3))
            raw_images.append(image_rawdata)
    return raw_images
