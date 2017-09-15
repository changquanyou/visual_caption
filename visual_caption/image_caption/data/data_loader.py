# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import ijson
import tensorflow as tf

from visual_caption.base.data.base_data_loader import BaseDataLoader
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig


# Data Loader class for AI_Challenge_2017

class ImageCaptionDataLoader(BaseDataLoader):
    def __init__(self, data_config):
        super().__init__(data_config=data_config)

    def _load_data(self, json_file, image_dir):
        batch_data = [self.data_config.batch_size]
        with open(json_file, mode='r', encoding='utf-8') as f:
            item_gen = ijson.items(f, "item")
            for item in enumerate(item_gen):
                batch_data.append(item)
                if len(batch_data) == self.data_config.batch_size:
                    yield batch_data
                    batch_data = []
            if len(batch_data) > 0:
                yield batch_data
            del batch_data

    def load_train_data(self):
        return self._load_data(json_file=self.data_config.train_json_data,
                               image_dir=self.data_config.train_image_dir)

    def load_test_data(self):
        return self._load_data(json_file=self.data_config.train_json_data)

    def load_validation_data(self):
        return self._load_data(json_file=self.data_config.validation_json_data,
                               image_dir=self.data_config.validation_image_dir)

    pass


def main(_):
    data_config = ImageCaptionDataConfig()
    data_loader = ImageCaptionDataLoader(data_config=data_config)
    data_gen = data_loader.load_validation_data()
    for batch, batch_data in enumerate(data_gen):
        print("batch={}".format(batch))
        for data in batch_data:
            print("data={}".format(data))


if __name__ == '__main__':
    tf.app.run()
