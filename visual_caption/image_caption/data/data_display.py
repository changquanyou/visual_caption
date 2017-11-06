# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import numpy as np
import skimage.io as io

from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader


class DataDisplay(object):
    """
    display data
    """

    def __init__(self):
        self.data_loader = ImageCaptionDataLoader()
        pass

    def display_image(self):
        data_gen = self.data_loader.load_data_generator(mode='train')
        for batch, batch_data in enumerate(data_gen):
            for data in batch_data:
                image_file = data['image_file']

                raw_image = io.imread(image_file)
                image_string = raw_image.tostring()

                decoded_image = np.fromstring(image_string, dtype=np.uint8)
                decoded_image = decoded_image.reshape(raw_image.shape)
                io.imshow(decoded_image)

                print("{}".format(np.allclose(raw_image, decoded_image)))


def main():
    display = DataDisplay()
    display.display_image()


if __name__ == '__main__':
    main()
