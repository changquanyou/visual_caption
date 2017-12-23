# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import sys

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from visual_caption.base.data.base_data_builder import BaseDataBuilder
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader
from visual_caption.image_caption.data.data_utils import ImageCaptionDataUtils
from visual_caption.image_caption.feature.feature_extractor import FeatureExtractor
from visual_caption.image_caption.feature.vgg_feature_manager import FeatureManager

class ImageCaptionDataBuilder(BaseDataBuilder):
    """
        Data Building:
        build train, test and validation dataset according with tfrecord format
    """

    def __init__(self, data_config):
        super(ImageCaptionDataBuilder, self).__init__(
            data_config=data_config)
        self.data_loader = ImageCaptionDataLoader()
        self.data_utils = ImageCaptionDataUtils()
        self.feature_extractor = FeatureExtractor()

    def _build_tfrecords(self, image_dir, data_gen, output_file):
        """
        convert AI_Challenge Image_Caption data to TFRecord file format with SequenceExample protos.
        """
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

            # if batch > 100:
            #     break

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
            # image_raw_data = self.data_utils.load_image_raw(image_filename)
            # image_batch.append(image_raw_data)
            image_batch.append(image_filename)

        # visual_features = self.feature_manager.get_vgg_feature(image_batch=image_batch)
        visual_features = self.feature_extractor.get_features(image_batch)

        # get list of sequence examples for batch caption_data
        for idx, caption_data in enumerate(batch_caption_data):  # for each caption meta data
            visual_feature = visual_features[idx]  #
            # get list of sequence examples for each caption_data
            example_list = self._to_sequence_example(
                visual_feature=visual_feature, caption_data=caption_data)
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
                self.data_config.visual_feature_name:
                    self._bytes_feature(visual_feature.tobytes())
            })

        # each image have multi caption texts
        # Extract the captions. Each image_id is associated with multiple captions.
        captions = caption_data['captions']
        sequence_example_list = []
        for caption in captions:  # for each caption text
            caption = str.strip(caption)
            if len(caption) > 0:
                caption_encoded = [char.encode() for char in caption]
                feature_lists = tf.train.FeatureLists(
                    feature_list={
                        self.data_config.caption_text_name:
                            self._bytes_feature_list(caption_encoded),
                    })

                sequence_example = tf.train.SequenceExample(
                    context=context, feature_lists=feature_lists)
                sequence_example_list.append(sequence_example)

        # assert len(sequence_example_list) == 5

        return sequence_example_list

    def build_train_data(self):
        image_dir = self.data_config.train_image_dir
        data_gen = self.data_loader.load_train_data()
        output_file = self.data_config.train_tf_data_file
        self._build_tfrecords(image_dir=image_dir,
                              data_gen=data_gen,
                              output_file=output_file)
        pass

    def build_test_data(self):
        image_dir = self.data_config.test_image_dir
        data_gen = self.data_loader.load_test_data()
        output_file = self.data_config.test_tf_data_file
        self._build_tfrecords(image_dir=image_dir,
                              data_gen=data_gen,
                              output_file=output_file)
        pass

    def build_valid_data(self):
        image_dir = self.data_config.valid_image_dir
        data_gen = self.data_loader.load_validation_data()
        output_file = self.data_config.valid_tf_data_file
        self._build_tfrecords(image_dir=image_dir,
                              data_gen=data_gen,
                              output_file=output_file)
        pass


def main(_):
    data_config = ImageCaptionDataConfig()
    data_builder = ImageCaptionDataBuilder(data_config=data_config)

    # data_builder.build_train_data()
    data_builder.build_valid_data()
    data_builder.build_test_data()


if __name__ == '__main__':
    tf.app.run()
