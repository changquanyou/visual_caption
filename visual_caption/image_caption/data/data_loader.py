# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import ijson.backends.yajl2_cffi as ijson

from visual_caption.base.data.base_data_loader import BaseDataLoader
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig

default_data_config = ImageCaptionDataConfig()
import tensorflow as tf


class ImageCaptionDataLoader(BaseDataLoader):
    """

    Data loader of raw data and prepare data for preprocessing

    Data Preprocessing:

        Prepare caption txt and token2vec model
    """

    def __init__(self, data_config=default_data_config):
        super(ImageCaptionDataLoader,self).__init__(data_config=data_config)
        # self.load_embeddings()

    def load_raw_generator(self, json_data_file, image_dir):
        """
        load json file and yield data in batch
        :param json_data_file:
        :param batch_size:
        :return:
        """
        batch_data = []

        # load_batch_size = self.data_config.batch_size
        load_batch_size = 80
        # count = 0;
        with open(json_data_file, mode='rb') as f_json:
            item_gen = ijson.items(f_json, "item")
            for item in enumerate(item_gen):
                (id, caption_dict) = item
                url = caption_dict['url']
                image_id = caption_dict['image_id']
                image_file = os.path.join(image_dir, image_id)
                captions = caption_dict['caption']
                caption_list = []
                for idx, caption_txt in enumerate(captions):
                    caption_txt = str.strip(caption_txt)
                    caption_txt = caption_txt.replace(' ', '')
                    caption_txt = caption_txt.replace('\n', '')
                    caption_list.append(caption_txt)

                caption_data = {
                    'id': id,
                    'url': url,
                    'image_id': image_id,
                    'image_file': image_file,
                    'captions': caption_list
                }
                batch_data.append(caption_data)

                if len(batch_data) == load_batch_size:
                    yield batch_data
                    batch_data = []

                    # count += 1
                    # if count % 1000 == 0:
                    #     print("load {} image data instances".format(count))

            if len(batch_data) > 0:
                yield batch_data
                del batch_data

    def load_test_data(self):
        return self.load_raw_generator(json_data_file=self.data_config.test_json_data,
                                       image_dir=self.data_config.test_image_dir)

    def load_train_data(self):
        return self.load_raw_generator(json_data_file=self.data_config.train_json_data,
                                       image_dir=self.data_config.train_image_dir)

    def load_validation_data(self):
        return self.load_raw_generator(json_data_file=self.data_config.validation_json_data,
                                       image_dir=self.data_config.validation_image_dir)


def main(_):
    data_loader = ImageCaptionDataLoader()
    # data_loader.build_char_all()
    data_loader.build_embeddings()


if __name__ == '__main__':
    tf.app.run()
