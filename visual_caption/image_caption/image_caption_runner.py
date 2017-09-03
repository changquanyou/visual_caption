# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf

from visual_caption.base.base_runner import BaseRunner
from  visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader
from visual_caption.image_caption.model.image_caption_config import ImageCaptionConfig
from visual_caption.image_caption.model.image_caption_model import ImageCaptionModel


class ImageCaptionRunner(BaseRunner):
    def __init__(self):
        data_config = ImageCaptionDataConfig(model_name="ImageCaption")
        data_loader = ImageCaptionDataLoader(data_config=data_config)
        model_config = ImageCaptionConfig(model_name=data_config.model_name)
        self.model = ImageCaptionModel(config=model_config, data_loader=data_loader)

    def train(self):
        self.model.run_train()
        pass

    def test(self):#
        self.model.run_test()
        pass

    def infer(self):
        self.model.run_infer()
        pass

    pass


def main(_):
    runner = ImageCaptionRunner()
    runner.train()


if __name__ == '__main__':
    tf.app.run()
