# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import sys

import tensorflow as tf

from visgen.feature.feature import FeatureManager
from visual_caption.base.data.base_data_builder import BaseDataBuilder
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader
from visual_caption.image_caption.data.data_utils import ImageCaptionDataUtils


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

        self.token2index = self.data_loader.token2index

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

        tf_writer = tf.python_io.TFRecordWriter(output_file)

        for batch, batch_data in enumerate(data_gen):  # for each batch

            # convert batch data to examples
            sequence_example_list = self._to_sequence_example_list(image_dir, batch_data)

            for sequence_example in sequence_example_list:
                if sequence_example is not None:
                    tf_writer.write(sequence_example.SerializeToString())

            if batch % 10 == 0 and batch > 0:
                print("flush batch {} dataset into file {}".format(batch, output_file))
                sys.stdout.flush()

            if batch % 100 == 0 and batch > 0:
                break

        tf_writer.close()
        sys.stdout.flush()

    def caption_to_ids(self, caption_txt):
        """
        not each token is not in token2index dict
        :param caption_txt:
        :return:
        """
        ids = [self.token2index[token] for token in caption_txt]
        return ids

    def _to_sequence_example_list(self, image_dir, batch_caption_data):
        """
        convert batch caption data to sequence example list
        :param batch_caption_data:
        :return:
        """
        sequence_example_list = []

        # for image batch
        image_batch = []
        for caption_data in batch_caption_data:
            image_id = caption_data['image_id']
            image_filename = os.path.join(image_dir, image_id)
            image_raw_data = self.data_utils.load_image_raw(image_filename)
            image_batch.append(image_raw_data)

        visual_features = self.feature_manager.get_vgg_feature(image_batch=image_batch)

        for idx, caption_data in enumerate(batch_caption_data):
            visual_feature = visual_features[idx]
            example_list = self._to_sequence_example(
                raw_image=image_batch[idx],
                visual_feature=visual_feature,
                caption_data=caption_data)
            sequence_example_list.extend(example_list)

        return sequence_example_list

    def _to_sequence_example(self, raw_image, visual_feature, caption_data):

        id = caption_data['id']  # instance id

        url = caption_data['url']
        encoded_url = url.encode()

        image_id = caption_data['image_id']
        encoded_image_id = image_id.encode()

        (height, width, depth) = raw_image.shape

        context = tf.train.Features(feature={

            "image/id": self._int64_feature(id),
            "image/url": self._bytes_feature(encoded_url),
            "image/image_id": self._bytes_feature(encoded_image_id),

            # "image/height": self._int64_feature(height),
            # "image/width": self._int64_feature(width),
            # "image/depth": self._int64_feature(depth),
            # "image/rawdata": self._bytes_feature(raw_image),

            # convert  ndarray to bytes for visual feature such as vgg19_fc7
            'image/visual_feature': self._bytes_feature(visual_feature.tobytes())
        })

        captions = caption_data['captions']
        sequence_example_list = []
        for caption_txt in captions:  # for each caption text
            encoded_caption_txt = [token.encode() for token in caption_txt.split()]
            caption_ids = self.caption_to_ids(caption_txt)

            feature_lists = tf.train.FeatureLists(feature_list={
                "image/caption": self._bytes_feature_list(encoded_caption_txt),
                "image/caption_ids": self._int64_feature_list(caption_ids)
            })

            sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
            sequence_example_list.append(sequence_example)

        return sequence_example_list


if __name__ == '__main__':
    data_builder = ImageCaptionDataBuilder()
    data_builder.build_tfrecords(mode='train')
