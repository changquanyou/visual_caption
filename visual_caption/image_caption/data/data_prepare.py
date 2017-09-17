# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import ijson
from gensim.models.word2vec import LineSentence, Word2Vec

from visual_caption.base.data.base_data_prepare import BaseDataPrepare
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig


class DataPrepare(BaseDataPrepare):
    """
    Prepare caption txt and token2vec model
    """

    def __init__(self):
        self.data_config = ImageCaptionDataConfig()

    def generate_char_txt(self):
        print("begin char txt generation ")
        self.build_char_text(json_data_file=self.data_config.train_json_data)
        self.build_char_text(json_data_file=self.data_config.validation_json_data)
        print("end char txt  generation")

    def build_char_text(self, json_data_file):
        print("begin char txt generation for {}".format(json_data_file))
        with open(json_data_file, mode='r', encoding='utf-8') as f_json, \
                open(file=self.data_config.caption_char_txt, mode='a', encoding='utf-8') as f_txt:
            item_gen = ijson.items(f_json, "item")
            for item in enumerate(item_gen):
                (id, caption_dict) = item
                captions = caption_dict['caption']
                for caption in captions:
                    if len(str.strip(caption)) > 0 :
                        line = [char + ' ' for char in caption]  # separate each token with a whitespace ' '
                        line[len(line) - 1] = '\n'  # replace the last token with '\n'
                        f_txt.writelines(line)
                if id % 10000 == 0:
                    print("Generating caption char txt for id={}".format(id))
        print("end char txt generation for {}".format(json_data_file))

        pass

    def generate_char2vec(self):
        sentences = LineSentence(self.data_config.caption_char_txt)
        dims = [50, 100, 200, 300]
        for dim_size in dims:
            model_file_name = "char2vec_" + str(dim_size) + ".model"
            model_file = os.path.join(self.data_config.embedding_dir, model_file_name)
            print("begin token2vec model {} generation".format(model_file))
            model = Word2Vec(sentences, size=dim_size, window=5, min_count=1, workers=4)
            model.save(model_file)
            print("Generated token2vec model to {}".format(model_file))

    def generate_word_file(self):
        pass


if __name__ == '__main__':
    data_prepare = DataPrepare()
    data_prepare.generate_char_txt()
    data_prepare.generate_char2vec()
