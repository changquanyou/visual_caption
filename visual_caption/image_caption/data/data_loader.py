# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import numpy as np
import tensorflow as tf
from gensim.models.word2vec import Word2Vec

from visual_caption.base.data.base_data_loader import BaseDataLoader
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig


# Data Loader class for AI_Challenge_2017

class ImageCaptionDataLoader(BaseDataLoader):
    """
        load train, validation, test dataset with embedding model

    """

    def __init__(self, data_config=ImageCaptionDataConfig()):
        super().__init__(data_config=data_config)
        self.tf_reader = tf.TFRecordReader()
        self._load_embeddings()

    def _load_embeddings(self):
        """
        load char2vec or word2vec model for token embeddings
        :return:
        """
        if not os.path.isfile(self.data_config.char2vec_model):
            self.build_embeddings()

        self.token2vec = Word2Vec.load(self.data_config.char2vec_model)
        self.vocab = self.token2vec.wv.vocab
        self.token2index = {}
        self.index2token = {}
        self.token_embedding_matrix = np.zeros([len(self.vocab) + 1, self.data_config.embedding_dim_size])
        for idx, token in enumerate(self.token2vec.wv.index2word):
            token_embedding = self.token2vec.wv[token]
            self.index2token[idx] = token
            self.token2index[token] = idx
            self.token_embedding_matrix[idx] = token_embedding
        self.token2index[self.data_config.unknown_token] = len(self.vocab)  # for unknown token
        pass

    def load_train_data(self):
        """
        load train data in batch and shuffe
        :return:
        """
        file_pattern = os.path.join(self.data_config.train_data_dir, '*.tfrecords')
        files = tf.train.match_filenames_once(file_pattern)
        filename_queue = tf.train.string_input_producer(files, shuffle=True)
        _, serialized_example = self.tf_reader.read(filename_queue)

        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features={
                'image/image_id': tf.FixedLenFeature([], dtype=tf.string),
                # 'image/height': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                # 'image/width': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                # 'image/channels': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                'image/rawdata': tf.FixedLenFeature([], dtype=tf.string)},
            sequence_features={
                'image/caption': tf.FixedLenSequenceFeature([], dtype=tf.string),
                'image/caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64)}
        )
        image_id = context['image/image_id']
        # image_height = context['image/height']
        # image_width = context['image/width']
        # image_channels = context['image/channels']
        image_rawdata = context['image/data']
        decoded_image = tf.decode_raw(image_rawdata, tf.uint8)
        # decoded_image = tf.reshape(decoded_image, [image_height, image_width, image_channels])

        image_caption = sequence['image/caption']
        image_caption_ids = sequence['image/caption_ids']

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * self.data_config.batch_size

        image_batch, caption_batch = tf.train.shuffle_batch(
            [image_id, decoded_image, image_caption, image_caption_ids],
            batch_size=self.data_config.batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue
        )
        yield image_caption, caption_batch

    def load_test_data(self):
        return self._load_data(json_file=self.data_config.train_json_data)

    def load_validation_data(self):
        return self._load_data(json_file=self.data_config.validation_json_data,
                               image_dir=self.data_config.validation_image_dir)

    pass


def main(_):
    data_loader = ImageCaptionDataLoader()
    data_gen = data_loader.load_train_data()
    for batch, batch_data in enumerate(data_gen):
        print("batch={}".format(batch))
        for data in batch_data:
            print("data={}".format(data))


if __name__ == '__main__':
    tf.app.run()
