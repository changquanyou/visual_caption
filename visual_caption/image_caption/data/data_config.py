# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

from visual_caption.base.data.base_data_config import BaseDataConfig


class ImageCaptionDataConfig(BaseDataConfig):
    def __init__(self):
        super().__init__()

        self.train_data_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_train_20170902")
        self.train_json_data = os.path.join(self.train_data_dir, "caption_train_annotations_20170902.json")
        self.train_image_dir = os.path.join(self.train_data_dir, "caption_train_images_20170902")

        self.validation_data_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_validation_20170910")
        self.validation_json_data = os.path.join(self.validation_data_dir,
                                                 "caption_validation_annotations_20170910.json")
        self.validation_image_dir = os.path.join(self.validation_data_dir, "caption_validation_images_20170910")

        self.caption_dir = os.path.join(self.model_data_dir, "captions")
        self.caption_char_txt = os.path.join(self.caption_dir, "caption_char.txt")
        self.caption_word_txt = os.path.join(self.caption_dir, "caption_word.txt")

        self.dim_size = 100
        self.embedding_dir = os.path.join(self.model_data_dir, "embeddings")
        char2vec_file_name = "char2vec_" + str(self.dim_size) + ".model"
        self.char2vec_model = os.path.join(self.embedding_dir, char2vec_file_name)

        word2vec_file_name = "word2vec_" + str(self.dim_size) + ".model"
        self.word2vec_model = os.path.join(self.embedding_dir, word2vec_file_name)

        self.seq_max_length = 100
    pass
