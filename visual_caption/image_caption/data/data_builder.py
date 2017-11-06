# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import sys

import tensorflow as tf

from visual_caption.base.data.base_data_builder import BaseDataBuilder
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader
from visual_caption.image_caption.data.data_utils import ImageCaptionDataUtils
from visual_caption.image_caption.feature.feature_manager import FeatureManager


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


class ImageCaptionDataBuilder(BaseDataBuilder):
    """
    Data Building:

        Data Builder of train, test and validation with tfrecord format

    """

    def __init__(self):

        self.data_config = ImageCaptionDataConfig()
        self.data_loader = ImageCaptionDataLoader()
        self.data_utils = ImageCaptionDataUtils()

        self.image_decoder = ImageDecoder()
        self.feature_manager = FeatureManager()

    def build_tfrecords(self, mode):
        """
        convert AI_Challenge Image_Caption data to TFRecord file format with SequenceExample protos.
        """

        if mode == 'train':
            image_dir = self.data_config.train_image_dir
            data_gen = self.data_loader.load_train_data()
            output_file = self.data_config.train_tf_data_file
        elif mode == 'test':
            image_dir = self.data_config.test_image_dir
            data_gen = self.data_loader.load_test_data()
            output_file = self.data_config.test_tf_data_file
        elif mode == 'validation':
            image_dir = self.data_config.validation_image_dir
            data_gen = self.data_loader.load_validation_data()
            output_file = self.data_config.validation_tf_data_file

        batch_length = 100  # each batch contains len(batch_data) data instances,
        for batch, batch_data in enumerate(data_gen):  # for each batch
            if batch % batch_length == 0:
                batch_begin = batch
                batch_end = batch_begin + batch_length
                file = output_file + "_" + str(batch_begin) + '_' + str(batch_end) + '.tfrecords'
                print("output_file = {}".format(file))
                tf_writer = tf.python_io.TFRecordWriter(file)

            sequence_example_list = self._to_sequence_example_list(image_dir, batch_data)
            for sequence_example in sequence_example_list:
                if sequence_example is not None:
                    tf_writer.write(sequence_example.SerializeToString())
            if batch % 10 == 0 and batch > 0:
                print("flush batch {} dataset into file {}".format(batch, file))
                sys.stdout.flush()
            if batch > 100:
                break

        tf_writer.close()
        sys.stdout.flush()

    def _to_sequence_example_list(self, image_dir, batch_caption_data):
        """
        convert batch caption data to sequence example list
        :param batch_caption_data:
        :return:
        """
        sequence_example_list = []

        # get visual features for image batch
        image_batch = []
        for caption_data in batch_caption_data:
            image_id = caption_data['image_id']
            image_filename = os.path.join(image_dir, image_id)
            image_raw_data = self.data_utils.load_image_raw(image_filename)
            image_batch.append(image_raw_data)
        visual_features = self.feature_manager.get_vgg_feature(image_batch=image_batch)

        # get list of sequence examples for batch caption_data
        for idx, caption_data in enumerate(batch_caption_data):  # for each caption meta data
            visual_feature = visual_features[idx]  #

            # get list of sequence examples for each caption_data
            example_list = self._to_sequence_example(visual_feature=visual_feature,
                                                     caption_data=caption_data)
            sequence_example_list.extend(example_list)

        return sequence_example_list

    def _to_sequence_example(self, visual_feature, caption_data):
        """
        convert caption meta data to tf sequence example
        :param visual_feature:
        :param caption_data:
        :return:
            list of tf sequence examples
        """
        id = caption_data['id']  # instance id
        url = caption_data['url']
        encoded_url = url.encode()

        # image_id is a string
        image_id = caption_data['image_id']
        encoded_image_id = image_id.encode()
        context = tf.train.Features(
            feature={
                "visual/id": self._int64_feature(id),
                "visual/url": self._bytes_feature(encoded_url),
                "visual/image_id": self._bytes_feature(encoded_image_id),
                # "image/rawdata": self._bytes_feature(raw_image),
                # convert  ndarray to bytes for visual feature, such as vgg19_fc7
                self.data_config.visual_feature_name: self._bytes_feature(visual_feature.tobytes())
            })

        # each image have multi caption texts
        # Extract the captions. Each image_id is associated with multiple captions.
        captions = caption_data['captions']
        sequence_example_list = []
        for caption in captions:  # for each caption text
            caption = str.strip(caption)
            if len(caption) > 0:
                line = [char for char in caption.strip()]  # separate each token with a whitespace
                line.insert(0, self.data_config.begin_token)
                line.append(self.data_config.end_token)
                caption_ids = [self.data_embeddings.token2index[char] for char in line]
                # # encoded_caption_
                caption_encoded = [char.encode() for char in caption]
                # ids for each caption text
                # caption_ids = self.caption_to_ids(caption)

                # feature list of each caption text and ids
                feature_lists = tf.train.FeatureLists(feature_list={
                    self.data_config.caption_text_name: self._bytes_feature_list(caption_encoded),
                    self.data_config.caption_ids_name: self._int64_feature_list(caption_ids)
                })

                sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
                sequence_example_list.append(sequence_example)

        assert len(sequence_example_list) == 5

        return sequence_example_list


if __name__ == '__main__':
    data_builder = ImageCaptionDataBuilder()
    # data_builder.build_tfrecords(mode='train')
    data_builder.build_tfrecords(mode='validation')
