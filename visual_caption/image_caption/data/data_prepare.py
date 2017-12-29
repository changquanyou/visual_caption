# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import tensorflow as tf

from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader
from visual_caption.utils.decorator_utils import timeit


class ImageCaptionDataPrepare(object):

    def __init__(self, data_config):
        self.data_config = data_config
        self.data_loader = ImageCaptionDataLoader(
            data_config=data_config)
        pass

    @timeit
    def build_vocabulary(self):
        token_set = set()
        token_set.add(self.data_config.token_start)
        token_set.add(self.data_config.token_end)
        token_set.add(self.data_config.token_unknown)
        token_set.add(self.data_config.token_pad)
        with open(file=self.data_config.caption_char_txt,
                  mode='r', encoding='utf-8') as f_char:
            captions = f_char.readlines()
            for caption in captions:
                for token in caption.split():
                    token_set.add(token)
        vocab_file = self.data_config.caption_vocab_txt
        with open(file=vocab_file, mode='w', encoding='utf-8') as f:
            for token in token_set:
                f.write(token + "\n")


    @timeit
    def build_char_all(self):
        """
        generate　Chinese chars txt file for the train and valid dataset
        Each sentence in test and train data is tokenized to Chinese char in per line.
        """
        if os.path.isfile(self.data_config.caption_char_txt):
            print("removing exist file {}".format(self.data_config.caption_char_txt))
            os.remove(self.data_config.caption_char_txt)
            print("exist file {} removed".format(self.data_config.caption_char_txt))
        self._build_char_text(
            json_data_file=self.data_config.train_json_data,
            image_dir=self.data_config.train_image_dir)
        self._build_char_text(
            json_data_file=self.data_config.valid_json_data,
            image_dir=self.data_config.valid_image_dir)

    @timeit
    def _build_char_text(self, json_data_file, image_dir):
        """
        build char sentence file, each sentence is separated by " "
        example:    <S> 这 是 一 个 例 子 </S>
        :param json_data_file:
        :return:
        """
        raw_data_gen = self.data_loader.load_raw_generator(
            json_data_file=json_data_file, image_dir=image_dir)
        with open(file=self.data_config.caption_char_txt, mode='a', encoding='utf-8') as f_txt:
            for batch, batch_data in enumerate(raw_data_gen):
                for raw_data in batch_data:
                    captions = raw_data['captions']
                    for caption in captions:
                        if len(str.strip(caption)) > 0:
                            line = [char + ' ' for char in caption]
                            # separate each token with a whitespace
                            line.insert(0, self.data_config.token_start + " ")
                            line.append(self.data_config.token_end)
                            line.append('\n')
                            f_txt.writelines(line)
                if batch % 1000 == 0 and batch > 0:
                    print("Generating caption char txt for batch={}".format(batch * 1000))
        pass


    pass


def main(_):
    data_config = ImageCaptionDataConfig()
    data_builder = ImageCaptionDataPrepare(data_config=data_config)
    data_builder.build_char_all()
    data_builder.build_vocabulary()
    pass


if __name__ == '__main__':
    tf.app.run()
