# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import numpy as np
import skimage.color
import skimage.io


def load_image(image_path):
    current_image = skimage.io.imread(image_path)
    # this is to convert a gray to RGB image
    if len(current_image.shape) == 2 or current_image.shape[2] == 1:
        current_image = skimage.color.gray2rgb(current_image)  #
    return current_image


def get_image_name(image_path):
    arrays = str(image_path).split('/')
    return arrays[len(arrays) - 1].split('.')[0]

def load_images(image_files):
    result = list()
    for image_file in image_files:
        current_image = load_image(image_file)
        result.append(current_image)
    return result


def crop_image(raw_image, xmin, ymin, width, height):
    """
    :param raw_image:
    :param x:
    :param y:
    :param width:
    :param height:
    :return:
        cropped_image
    """
    xmin = int(xmin)
    ymin = int(ymin)

    xmax = int(xmin + width)
    ymax = int(ymin + height)

    crop_im = np.zeros(raw_image.shape)
    crop_im[xmin:xmax, ymin:ymax, :] = raw_image[xmin:xmax, ymin:ymax, :]
    cropped_shape = crop_im.shape
    if cropped_shape[0] <= 0 or cropped_shape[1] <= 0:
        print("raw_shape={}, cropped_shape={}".format(raw_image.shape, cropped_shape))
        crop_im = None
    return crop_im
