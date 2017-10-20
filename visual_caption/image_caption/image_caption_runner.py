# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf

from visual_caption.base.base_runner import BaseRunner
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_reader import ImageCaptionDataReader
from visual_caption.image_caption.model.image_caption_config import ImageCaptionConfig
from visual_caption.image_caption.model.image_caption_model import ImageCaptionModel


class ImageCaptionRunner(BaseRunner):
    def __init__(self):
        data_config = ImageCaptionDataConfig()
        data_reader = ImageCaptionDataReader(data_config=data_config)
        model_config = ImageCaptionConfig(model_name=data_config.model_name, mode='train')
        self.model = ImageCaptionModel(config=model_config, data_reader=data_reader)

    def train(self):
        self.model.run_train()
        pass

    def test(self):  #
        self.model.run_test()
        pass

    def inference(self):
        self.model.run_infer()
        pass

    pass


def main(_):
    runner = ImageCaptionRunner()
    runner.train()


if __name__ == '__main__':
    tf.app.run()
