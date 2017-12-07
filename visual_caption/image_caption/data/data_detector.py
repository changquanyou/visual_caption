# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import json
import os
import time

import tensorflow as tf

from tf_visgen.mscoco.data.mscoco_data_config import MSCOCODataConfig
from tf_visgen.mscoco.data.mscoco_data_loader import MSCOCODataLoader
from tf_visgen.utils.decorator_utils import timeit
from tf_visgen.visgen.feature.faster_rcnn_detector import FasterRCNNDetector, DetectorConfig

detector_config = DetectorConfig()
faster_rcnn_detector = FasterRCNNDetector(detector_config)


def get_image_id(image_filepath):
    idx_start =  image_filepath.rfind('_')
    idx_end  = image_filepath.rfind('.')
    id_string = image_filepath[idx_start+1:idx_end]
    image_id = int(id_string)
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
        self.data_loader = MSCOCODataLoader()
        self.detector = faster_rcnn_detector

    @timeit
    def build_bbox_data(self, caption_file, image_dir, target_file):
        metadata = self.data_loader.load_metadata(
            captions_file=caption_file, image_dir=image_dir)
        # convert metadata of each image to bbox record
        image_list = list()  # batch data for images
        start = time.time()
        for idx, image_data in enumerate(metadata):
            image_filename = image_data.filename
            image_dict = self.detector.detect_image(
                image_path=image_filename)
            image_dict["image_id"] = image_data.image_id
            image_dict["image_filepath"] = image_data.filename
            image_dict["captions"] = image_data.captions
            image_list.append(image_dict)
            length = len(image_list)
            if length % 100 == 0:
                elapsed = time.time() - start
                print("detected {} images for {}, elapsed {}."
                      .format(length, caption_file, elapsed))
            if idx >= 1000:  # for partial data
                break
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
            print("dumped metadata from {} to {}".format(test_image_dir, target_file))

    @timeit
    def build_train_data(self):
        image_dir = self.data_config.train_image_dir
        caption_file = self.data_config.captions_train_file
        detect_file = self.data_config.detect_train_file
        self.build_bbox_data(caption_file=caption_file,
                             image_dir=image_dir,
                             target_file=detect_file)

    @timeit
    def build_valid_data(self):
        image_dir = self.data_config.valid_image_dir
        caption_file = self.data_config.captions_valid_file
        detect_file = self.data_config.detect_valid_file
        self.build_bbox_data(caption_file=caption_file,
                             image_dir=image_dir,
                             target_file=detect_file)

    @timeit
    def build_test_data(self):
        image_dir = self.data_config.test_image_dir
        detect_file = self.data_config.detect_test_file
        self.build_bbox_test_data(test_image_dir=image_dir,
                                  target_file=detect_file)

    @timeit
    def build_all_bbox(self):
        # self.build_train_data()
        # self.build_valid_data()
        self.build_test_data()


def main(_):
    data_config = MSCOCODataConfig()
    data_detector = COCODataDetector(data_config)
    data_detector.build_all_bbox()


if __name__ == '__main__':
    tf.app.run()
