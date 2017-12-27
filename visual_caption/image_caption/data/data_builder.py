# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import sys

import ijson
import tensorflow as tf
import time
from tensorflow.contrib.learn import ModeKeys

from visual_caption.base.data.base_data_builder import BaseDataBuilder
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader
from visual_caption.image_caption.feature.feature_extractor import FeatureExtractor
from visual_caption.utils import image_utils
from visual_caption.utils.decorator_utils import timeit

import numpy as np
from object_detection.utils import dataset_util


class ImageCaptionDataBuilder(BaseDataBuilder):
    """
        Data Building:
        build train, test and validation dataset according with tfrecord format
    """

    def __init__(self, data_config):
        super(ImageCaptionDataBuilder, self).__init__(data_config)
        # visual feature extractor based on inception_resnet_v2
        self.data_loader = ImageCaptionDataLoader(data_config=data_config)
        self.feature_extractor = None
        pass

    def _to_tf_example(self, mode, image_data):
        """
        Convert python dictionary format data of one image to tf.Example proto.
        Args:
            image_data: information of one image, include
                bounding box, labels of bounding box,
                height, width, encoded pixel data.
        Returns:
            example: The converted tf.Example
        """
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
        image_id = image_data['image_id']
        image_path = image_data['image_file']
        image_raw_data = image_utils.load_image(image_path)
        (image_height, image_width, image_depth) = image_raw_data.shape
        image_feature = self.feature_extractor.get_feature_from_rawdata(image_raw_data)

        bboxes = image_data['bboxes']
        bbox_number = len(bboxes)
        max_bbox_number = self.data_config.num_max_bbox
        if bbox_number > max_bbox_number:
            bbox_number = max_bbox_number
        bbox_raw_data_list = list()
        bbox_labels_ids = list()
        bbox_labels_names = list()

        bbox_list = list()
        for idx in range(bbox_number):
            bbox = bboxes[idx]
            xmin = bbox['x_min']
            ymin = bbox['y_min']
            xmax = bbox['x_max']
            ymax = bbox['y_max']

            bbox_list.append([xmin, ymin, xmax, ymax])
            bbox_labels_ids.append(bbox['class_id'])
            bbox_labels_names.append(bbox['class_name'])

            x_width = xmax - xmin
            y_height = ymax - ymin
            bbox_raw_data = image_utils.crop_image(
                image_raw_data, xmin=xmin, ymin=ymin,
                width=x_width, height=y_height)
            bbox_raw_data_list.append(bbox_raw_data)

        bboxes_data = np.reshape(bbox_list, (-1))
        bbox_features = self.feature_extractor.get_feature_from_rawdata_list(
            bbox_raw_data_list)
        bbox_features = np.asarray(bbox_features)
        # image_id_encoded = [char.encode() for char in image_id]
        tf_context = tf.train.Features(feature={
            # image data
            'image/image_id': dataset_util.bytes_feature(str.encode(image_id)),
            'image/height': dataset_util.int64_feature(image_height),
            'image/width': dataset_util.int64_feature(image_width),
            'image/depth': dataset_util.int64_feature(image_depth),
            'image/feature': dataset_util.bytes_feature(image_feature.tobytes()),

            # bbox
            'bbox/number': dataset_util.int64_feature(bbox_number),
            'bbox/labels': dataset_util.int64_list_feature(bbox_labels_ids),
            'bbox/bboxes': dataset_util.int64_list_feature(bboxes_data),
            'bbox/features': dataset_util.bytes_feature(bbox_features.tobytes())
        })

        tf_example_list = list()
        captions = image_data['captions']
        for caption in captions:
            caption_encoded = [char.encode() for char in caption]
            feature_lists = tf.train.FeatureLists(feature_list={
                'caption': self._bytes_feature_list(caption_encoded), })
            tf_example = tf.train.SequenceExample(
                context=tf_context, feature_lists=feature_lists)
            tf_example_list.append(tf_example)
        return tf_example_list

    def _write_tf_examples(self, tf_writer, tf_batch):
        for tf_example in tf_batch:
            tf_writer.write(tf_example.SerializeToString())
        print("saved {} tf_records".format(len(tf_batch)))
        pass

    def _build_tfrecords(self, data_mode, output_dir=None, detected_data_file=None):
        """
        convert detected_data_file to tfrecords
        :param detected_data_file: bboxes data file for images
        :param tfrecord_file:
        :return:
        """
        shard_size = 1000
        time_begin = time.time()
        shard_data = []
        count = 0
        start = 1

        with open(file=detected_data_file, mode='rb') as f:
            items = ijson.items(f, "item")
            for image_data in items:
                # convert each image data into tf_example list
                tf_example_list = self._to_tf_example(
                    mode=ModeKeys.INFER, image_data=image_data)
                for tf_example in tf_example_list:
                    shard_data.append(tf_example)
                count += 1
                if count % 10 == 0:
                    print("build {} image data".format(count))
                if count % shard_size == 0:  # save shard data and change shard tf_writer
                    tfrecord_file_name = data_mode + "_data" + "_" + str(start) + "_" + str(count) + '.tfrecords'
                    tfrecord_file = os.path.join(output_dir, tfrecord_file_name)
                    tf_writer = tf.python_io.TFRecordWriter(tfrecord_file)
                    self._write_tf_examples(tf_writer, shard_data)
                    sys.stdout.flush()
                    print("converted image data {}-{} into {}, elapsed {} sec."
                          .format(start, count, tfrecord_file, time.time() - time_begin))
                    shard_data = []
                    time_begin = time.time()
                    start = count + 1

            if len(shard_data) > 0:
                tfrecord_file_name = data_mode + "_data" + "_" + str(start) + "_" + str(count) + '.tfrecords'
                tfrecord_file = os.path.join(output_dir, tfrecord_file_name)
                tf_writer = tf.python_io.TFRecordWriter(tfrecord_file)
                self._write_tf_examples(tf_writer, shard_data)
                sys.stdout.flush()
                print("converted image data {}-{} into {}, elapsed {} sec."
                      .format(start, count, tfrecord_file, time.time() - time_begin))
        pass

    def build_train_data(self):
        detect_file = self.data_config.detect_train_file
        output_dir = self.data_config.train_data_dir
        self._build_tfrecords(data_mode=ModeKeys.TRAIN,
                              detected_data_file=detect_file,
                              output_dir=output_dir)

    def build_valid_data(self):
        detect_file = self.data_config.detect_valid_file
        output_dir = self.data_config.valid_data_dir
        self._build_tfrecords(data_mode=ModeKeys.TRAIN,
                              detected_data_file=detect_file,
                              output_dir=output_dir)
        pass

    def build_test_data(self):
        detect_file = self.data_config.detect_test_file
        output_dir = self.data_config.test_data_dir
        self._build_tfrecords(data_mode=ModeKeys.INFER,
                              detected_data_file=detect_file,
                              output_dir=output_dir)
        pass

    @timeit
    def build_vocabulary(self):
        token_set = set()
        token_set.add(self.data_config.token_start)
        token_set.add(self.data_config.token_end)
        token_set.add(self.data_config.token_unknown)
        token_set.add(self.data_config.token_pad)
        with open(file=self.data_config.caption_char_txt,
                  mode='r', encoding='utf-8') as f_char:
            captions = f_char.readlines()
            for caption in captions:
                for token in caption.split():
                    token_set.add(token)
        vocab_file = self.data_config.vocab_file
        with open(file=vocab_file, mode='w', encoding='utf-8') as f:
            for token in token_set:
                f.write(token + "\n")

    @timeit
    def _build_char_text(self, json_data_file, image_dir):
        """
        build char sentence file, each sentence is separated by " "
        example:    <S> 这 是 一 个 例 子 </S>
        :param json_data_file:
        :return:
        """
        raw_data_gen = self.data_loader.load_raw_generator(
            json_data_file=json_data_file, image_dir=image_dir)
        with open(file=self.data_config.caption_char_txt, mode='a', encoding='utf-8') as f_txt:
            for batch, batch_data in enumerate(raw_data_gen):
                for raw_data in batch_data:
                    captions = raw_data['captions']
                    for caption in captions:
                        if len(str.strip(caption)) > 0:
                            line = [char + ' ' for char in caption]
                            # separate each token with a whitespace
                            line.insert(0, self.data_config.token_start + " ")
                            line.append(self.data_config.token_end)
                            line.append('\n')
                            f_txt.writelines(line)
                if batch % 1000 == 0:
                    print("Generating caption char txt for batch={}".format(batch * 1000))
        pass

    @timeit
    def build_char_all(self):
        """ generate　Chinese chars txt file for the train and valid dataset
            Each sentence in test and train data is tokenized to Chinese char in per line.
        """
        self._build_char_text(
            json_data_file=self.data_config.train_json_data,
            image_dir=self.data_config.train_image_dir)
        self._build_char_text(
            json_data_file=self.data_config.valid_json_data,
            image_dir=self.data_config.valid_image_dir)

    pass


def main(_):
    data_config = ImageCaptionDataConfig()
    data_builder = ImageCaptionDataBuilder(data_config=data_config)
    # data_builder.build_char_all()
    # data_builder.build_vocabulary()

    data_builder.build_all_data()


if __name__ == '__main__':
    tf.app.run()
