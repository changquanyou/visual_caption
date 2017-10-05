# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import numpy as np
import tensorflow as tf
from gensim.models.word2vec import Word2Vec

from visual_caption.base.data.base_data_reader import BaseDataReader
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig

# Data Reader class for AI_Challenge_2017

context_feature = {
    'image/id': tf.FixedLenFeature([], dtype=tf.int64),
    'image/url': tf.FixedLenFeature([], dtype=tf.string),
    'image/image_id': tf.FixedLenFeature([], dtype=tf.string),
    # 'image/height': tf.FixedLenFeature([], dtype=tf.int64),
    # 'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    # 'image/depth': tf.FixedLenFeature([], dtype=tf.int64),
    # 'image/rawdata': tf.FixedLenFeature([], dtype=tf.string)
}

sequence_features = {
    'image/caption': tf.FixedLenSequenceFeature([], dtype=tf.string),
    'image/caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64)
}


class ImageCaptionDataReader(BaseDataReader):
    """
        Read data for train, validation, test dataset with embedding model
    """

    def __init__(self, data_config=ImageCaptionDataConfig()):
        super().__init__(data_config=data_config)
        self.tf_reader = tf.TFRecordReader()
        self._load_embeddings()

    def load_train_data(self):
        """
        load train data in batch and shuffe
        :return:
        """
        file_pattern = os.path.join(self.data_config.train_data_dir, '*.tfrecords')
        files = tf.train.match_filenames_once(file_pattern)
        filename_queue = tf.train.string_input_producer(files, shuffle=True)
        _, serialized_example = self.tf_reader.read(filename_queue)

    def load_test_data(self):
        return self._load_data(json_file=self.data_config.train_json_data)

    def load_validation_data(self):
        return self._load_data(json_file=self.data_config.validation_json_data,
                               image_dir=self.data_config.validation_image_dir)

    def parse_single_sequence_example(self, serialized_example):
        # parsing sequence example
        context, sequence = tf.parse_single_sequence_example(
            serialized_example, context_features=context_feature, sequence_features=sequence_features
        )

        image_id = context['image/image_id']
        height = tf.cast(context['image/height'], tf.int64)
        width = tf.cast(context['image/width'], tf.int64)
        depth = tf.cast(context['image/depth'], tf.int64)
        image_rawdata = context['image/rawdata']

        reshaped_image = tf.reshape(image_rawdata, tf.stack([height, width, depth]))

        image_caption = sequence['image/caption']
        image_caption_ids = sequence['image/caption_ids']

        caption_length = tf.shape(image_caption_ids)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

        input_seq = tf.slice(image_caption_ids, [0], input_length)
        target_seq = tf.slice(image_caption_ids, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)

        return [image_id, reshaped_image, input_seq, target_seq, indicator]

    def read_tfrecords(self):

        file_pattern = os.path.join(self.data_config.train_data_dir, '*.tfrecords')
        files = tf.train.match_filenames_once(file_pattern)
        filename_queue = tf.train.string_input_producer(files, shuffle=True)

        _, serialized_example = self.tf_reader.read(filename_queue)

        single_data = self.parse_single_sequence_example(serialized_example=serialized_example)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * self.data_config.batch_size

        batch_data = tf.train.shuffle_batch(single_data,
                                            batch_size=self.data_config.batch_size, capacity=capacity,
                                            min_after_dequeue=min_after_dequeue)

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session(config=self.sess_config) as sess:
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                for batch_index in range(5):
                    image_id_batch, reshaped_image_batch, input_seq_batch, target_seq_batch, indicator_batch = sess.run(
                        [batch_data])
                    for idx, img_id in enumerate(image_id_batch):
                        print("image_id={},input_seq={},reshaped_image={}".format(img_id, input_seq_batch[idx],
                                                                                  reshaped_image_batch[idx]))

            except Exception as e:
                print(e)
                coord.request_stop(e)
            finally:
                coord.request_stop()  # Stop the threads
                coord.join(threads)  # Wait for threads to stop


def main(_):
    data_reader = ImageCaptionDataReader()


if __name__ == '__main__':
    tf.app.run()
