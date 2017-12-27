# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import json
import os
import time

import tensorflow as tf

from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader
from visual_caption.utils.decorator_utils import timeit
from visual_caption.image_caption.feature.faster_rcnn_detector import FasterRCNNDetector, DetectorConfig

detector_config = DetectorConfig()
faster_rcnn_detector = FasterRCNNDetector(detector_config)


def get_image_id(image_filepath):
    idx_start = image_filepath.rfind('_')
    idx_end = image_filepath.rfind('.')
    image_id = image_filepath[idx_start + 1:idx_end]
    # image_id = int(id_string)
    return image_id


class ImageCaptionDataDetector(object):
    """
    a special data builder for mscoco data

    in order to detect the regions of objects in mscoco image
    convert coco metadata to metadata with bboxes
    saved to json data
    """

    def __init__(self, data_config):
        self.data_config = data_config
        self.data_loader = ImageCaptionDataLoader()
        self.detector = faster_rcnn_detector

    @timeit
    def build_bbox_data(self, caption_file, image_dir, target_file):
        data_gen = self.data_loader.load_raw_generator(
            json_data_file=caption_file, image_dir=image_dir)
        # convert metadata of each image to bbox record
        image_list = list()  # batch data for images
        start = time.time()
        for batch, batch_data in enumerate(data_gen):
            for idx, image_data in enumerate(batch_data):
                image_filename =image_data['image_file']
                image_dict = self.detector.detect_image(
                    image_path=image_filename)
                image_dict['id'] = image_data['id']
                image_dict['url'] = image_data['url']
                image_dict["image_id"] = image_data['image_id']
                image_dict["image_file"] = image_data['image_file']
                image_dict["captions"] = image_data['captions']

                image_list.append(image_dict)
                length = len(image_list)
                if length % 100 == 0:
                    elapsed = time.time() - start
                    print("detected {} images for {}, elapsed {}."
                          .format(length, caption_file, elapsed))
            # if batch >= 10:  # for partial data
            #     break
        with open(target_file, 'w') as fp:
            json.dump(image_list, fp=fp, sort_keys=True)
            print("dumped metadata from {} to {}".format(caption_file, target_file))

    def build_bbox_test_data(self, test_image_dir, target_file):
        image_list = list()  # batch data for images
        start = time.time()
        image_files = os.listdir(test_image_dir)
        for idx, image_file in enumerate(image_files):
            image_file_path = os.path.join(test_image_dir, image_file)
            image_dict = self.detector.detect_image(image_path=image_file_path)
            image_id = get_image_id(image_file)

            image_dict["image_id"] = image_id
            image_dict["image_filepath"] = image_file_path
            image_dict["captions"] = ["This is a test caption text"]
            image_list.append(image_dict)
            length = len(image_list)
            if length % 10 == 0:
                elapsed = time.time() - start
                print("detected {} images for {}, elapsed {}."
                      .format(length, test_image_dir, elapsed))
            if idx >= 1000:  # for partial data
                break

        with open(target_file, 'w') as fp:
            json.dump(image_list, fp=fp, sort_keys=True)
            print("dumped metadata from {} to {}".format(
                test_image_dir, target_file))

    @timeit
    def build_train_data(self):
        image_dir = self.data_config.train_image_dir
        caption_file = self.data_config.train_json_data
        detect_file = self.data_config.detect_train_file
        self.build_bbox_data(
            caption_file=caption_file,
            image_dir=image_dir,
            target_file=detect_file)

    @timeit
    def build_valid_data(self):
        image_dir = self.data_config.valid_image_dir
        caption_file = self.data_config.valid_json_data
        detect_file = self.data_config.detect_valid_file
        self.build_bbox_data(
            caption_file=caption_file,
            image_dir=image_dir,
            target_file=detect_file)

    @timeit
    def build_test_data(self):
        image_dir = self.data_config.test_image_dir
        detect_file = self.data_config.detect_test_file
        self.build_bbox_test_data(
            test_image_dir=image_dir, target_file=detect_file)

    @timeit
    def build_all_bbox(self):
        # self.build_train_data()
        # self.build_valid_data()
        self.build_test_data()


def main(_):
    data_config = ImageCaptionDataConfig()
    data_detector = ImageCaptionDataDetector(data_config)
    data_detector.build_all_bbox()


if __name__ == '__main__':
    tf.app.run()
