# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

from visual_caption.base.data.base_data_config import BaseDataConfig

BEGIN_TOKEN = "<S>"
END_TOKEN = "<S>"
UNKNOWN_TOKEN = "#UNKNOWN#"


class ImageCaptionDataConfig(BaseDataConfig):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        super().__init__()

        # for train
        self.train_rawdata_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_train_20170902")
        self.train_json_data = os.path.join(self.train_rawdata_dir, "caption_train_annotations_20170902.json")
        self.train_image_dir = os.path.join(self.train_rawdata_dir, "caption_train_images_20170902")
        self.train_data_dir = os.path.join(self.model_data_dir, "train")
        self.train_tf_data_file = os.path.join(self.train_data_dir, "train.tfrecords")

        # for validation
        self.validation_rawdata_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_validation_20170910")
        self.validation_json_data = os.path.join(self.validation_rawdata_dir,
                                                 "caption_validation_annotations_20170910.json")
        self.validation_image_dir = os.path.join(self.validation_rawdata_dir, "caption_validation_images_20170910")
        self.validation_data_dir = os.path.join(self.model_data_dir, "validation")
        self.validation_tf_data_file = os.path.join(self.validation_data_dir, "validation.tfrecords")

        # for test
        self.test_rawdata_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_test_20170910")
        self.test_json_data = os.path.join(self.test_rawdata_dir, "caption_test_annotations_20170910.json")
        self.test_image_dir = os.path.join(self.test_rawdata_dir, "caption_test_images_20170910")
        self.test_data_dir = os.path.join(self.model_data_dir, "test")
        self.test_tf_data_file = os.path.join(self.test_data_dir, "test.tfrecords")

        # for caption txt data
        self.caption_dir = os.path.join(self.model_data_dir, "captions")
        self.caption_char_txt = os.path.join(self.caption_dir, "caption_char.txt")
        self.caption_word_txt = os.path.join(self.caption_dir, "caption_word.txt")

        # for embeddings
        self.embedding_dim_size = 100  # default dim size
        self.embedding_dir = os.path.join(self.model_data_dir, "embeddings")
        char2vec_file_name = "char2vec_" + str(self.embedding_dim_size) + ".model"
        self.char2vec_model = os.path.join(self.embedding_dir, char2vec_file_name)
        word2vec_file_name = "word2vec_" + str(self.embedding_dim_size) + ".model"
        self.word2vec_model = os.path.join(self.embedding_dir, word2vec_file_name)

        # for seq2seq model
        self.seq_max_length = 100
        self.unknown_token = UNKNOWN_TOKEN
        self.begin_token = BEGIN_TOKEN
        self.end_token = END_TOKEN


        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"

        # Name of the SequenceExample feature list containing integer captions.
        self.caption_feature_name = "image/caption_ids"


    pass
