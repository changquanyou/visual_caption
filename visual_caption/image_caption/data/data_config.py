# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

from visual_caption.base.data.base_data_config import BaseDataConfig

TOKEN_BEGIN = "<S>"
TOKEN_END = "</S>"
TOKEN_UNKNOWN = "<UNKNOWN>"

MODEL_NAME = 'image_caption'
MODE = 'train'


class ImageCaptionDataConfig(BaseDataConfig):
    """Wrapper class for model hyperparameters."""

    def __init__(self, mode=MODE, model_name=MODEL_NAME):
        super(ImageCaptionDataConfig,self).__init__(
            mode=mode, model_name=model_name)

        # for train
        self.train_rawdata_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_train_20170902")
        self.train_json_data = os.path.join(self.train_rawdata_dir, "caption_train_annotations_20170902.json")
        self.train_image_dir = os.path.join(self.train_rawdata_dir, "caption_train_images_20170902")
        self.train_data_dir = os.path.join(self.model_data_dir, "train")
        self.train_tf_data_file = os.path.join(self.train_data_dir, "image_caption_train")

        # for validation
        self.validation_rawdata_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_validation_20170910")
        self.validation_json_data = os.path.join(self.validation_rawdata_dir,
                                                 "caption_validation_annotations_20170910.json")
        self.validation_image_dir = os.path.join(self.validation_rawdata_dir, "caption_validation_images_20170910")
        self.validation_data_dir = os.path.join(self.model_data_dir, "valid")
        self.validation_tf_data_file = os.path.join(self.validation_data_dir, "image_caption_validation")

        # for test
        self.test_rawdata_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_test1_20170923")
        self.test_json_data = os.path.join(self.test_rawdata_dir, "caption_test_annotations_20170910.json")
        self.test_image_dir = os.path.join(self.test_rawdata_dir, "caption_test1_images_20170923")
        self.test_data_dir = os.path.join(self.model_data_dir, "test")
        self.test_tf_data_file = os.path.join(self.test_data_dir, "image_caption_test")

        # for caption txt data
        self.caption_dir = os.path.join(self.model_data_dir, "captions")
        self.caption_char_txt = os.path.join(self.caption_dir, "caption_char.txt")
        self.caption_word_txt = os.path.join(self.caption_dir, "caption_word.txt")

        # for embeddings
        self.embedding_dim_size = 300  # default dim size
        self.embedding_dir = os.path.join(self.model_data_dir, "embeddings")
        char2vec_file_name = "char2vec_" + str(self.embedding_dim_size) + ".model"
        self.char2vec_model = os.path.join(self.embedding_dir, char2vec_file_name)
        word2vec_file_name = "word2vec_" + str(self.embedding_dim_size) + ".model"
        self.word2vec_model = os.path.join(self.embedding_dir, word2vec_file_name)

        self.vocab_file = os.path.join(self.embedding_dir, "vocab.txt")

        # for encoder-decoder model
        self.seq_max_length = 100
        self.token_unknown = TOKEN_UNKNOWN
        self.token_begin = TOKEN_BEGIN
        self.token_end = TOKEN_END

        # for tfrecord format data
        # Name of the SequenceExample context feature containing image data.
        self.visual_feature_name = "visual/feature"
        # Name of the SequenceExample context feature containing image data.
        self.visual_image_id_name = "visual/image_id"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_text_name = "caption/text"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_ids_name = "caption/ids"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_target_ids__name = "caption/target_ids"

        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None
        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 2300
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 1
        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

    pass
