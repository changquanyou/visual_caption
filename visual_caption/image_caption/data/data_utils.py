# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import skimage.color
import skimage.io


class ImageCaptionDataUtils(object):
    def load_embeddings(self):
        pass

    @staticmethod
    def load_image_raw(image_file):
        """
        load image data base on given image id
        :param image_id: image id
        :return: image data
        """
        current_image = skimage.io.imread(image_file)
        if len(current_image.shape) == 2 or current_image.shape[2] == 1:  # this is to convert a gray to RGB image
            current_image = skimage.color.gray2rgb(current_image)  #
        return current_image
