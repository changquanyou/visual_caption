# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

from gensim.models.word2vec import LineSentence, Word2Vec

from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader


class DataUtils(object):
    def __init__(self):
        self.data_config = ImageCaptionDataConfig()
        self.data_loader = ImageCaptionDataLoader(data_config=self.data_config)

    def generate_char_file(self):
        train_gen = self.data_loader.load_train_data()
        with (open(file=self.data_config.caption_char_txt, mode='a', encoding='utf-8')) as f:
            for batch, batch_data in enumerate(train_gen):
                for caption_data in batch_data:
                    (id, dict) = caption_data
                    captions = dict['caption']
                    for caption in captions:
                        line = [char + ' ' for char in caption]
                        line.append('\n')
                        f.writelines(line)

                    if id % 1000 == 0:
                        print("Generating caption char txt for id={}".format(id))

        pass

    def generate_char2vec(self):
        sentences = LineSentence(self.data_config.caption_char_txt)
        dims = [50, 100, 200, 300]
        for dim_size in dims:
            model = Word2Vec(sentences, size=dim_size, window=5, min_count=0, workers=4)
            model_file_name = "char2vec" + str(dim_size) + ".model"
            model_file = os.path.join(self.data_config.embedding_dir, model_file_name)
            model.save(model_file)

    def generate_word_file(self):
        pass


if __name__ == '__main__':
    data_utils = DataUtils()
    data_utils.generate_char2vec()
