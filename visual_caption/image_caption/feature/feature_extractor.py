# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from pathlib import Path

import tensorflow as tf
from scipy.misc import imresize
from sklearn.preprocessing import normalize

from visual_caption.utils import image_utils
from visual_caption.utils.decorator_utils import timeit

slim = tf.contrib.slim
from slim.nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

home = str(Path.home())  # home dir
base_data_dir = os.path.join(home, 'data')
model_data_dir = os.path.join(base_data_dir, "tf/models")
inception_resnet_v2_ckpt = os.path.join(model_data_dir, "inception_resnet_v2_2016_08_30.ckpt")

batch_size = 40


def load_images(image_files):
    raw_images = list()
    for idx, image_path in enumerate(image_files):
        image_rawdata = image_utils.load_image(image_path=image_path)
        if image_rawdata is not None:
            image_rawdata = imresize(image_rawdata, (299, 299, 3))
            raw_images.append(image_rawdata)
    return raw_images


class FeatureExtractor(object):
    """
    inception_resnet_v2 Feature Extractor
    """

    def __init__(self, sess=None):
        self.input_images = tf.placeholder(
            tf.float32,
            shape=(None, 299, 299, 3),
            name='input_images')
        if sess is None:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=0.6)
            sess_config = tf.ConfigProto(
                gpu_options=gpu_options,
                allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=sess_config)
        else:
            self.sess = sess
        self._load_model()

    @timeit
    def _load_model(self):
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), self.input_images)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_resnet_v2(
                scaled_input_tensor, is_training=False)
        saver = tf.train.Saver()
        checkpoint_file = inception_resnet_v2_ckpt
        saver.restore(self.sess, checkpoint_file)
        self.fetches = [end_points['PreLogitsFlatten'], logits]

    def get_features(self, image_files):
        raw_images = load_images(image_files)
        feed_dict = {self.input_images: raw_images}
        predict_values, logit_values = self.sess.run(
            self.fetches, feed_dict)
        results = normalize(predict_values)
        return results

    def get_feature(self, image_path):
        image_files = [image_path]
        features = self.get_features(image_files)
        return features[0]

    def get_feature_from_rawdata(self, image_raw_data):
        image_rawdata_list = [image_raw_data]
        return self.get_feature_from_rawdata_list(image_rawdata_list)[0]

    def get_feature_from_rawdata_list(self, image_rawdata_list):
        raw_image_datas = list()
        results = list()
        for image_rawdata in image_rawdata_list:
            image_rawdata = imresize(image_rawdata, (299, 299, 3))
            raw_image_datas.append(image_rawdata)

            if len(raw_image_datas) == batch_size:
                feed_dict = {self.input_images: raw_image_datas}
                predict_values, logit_values = self.sess.run(
                    self.fetches, feed_dict)
                results.extend(normalize(predict_values))
                raw_image_datas = list()

        if len(raw_image_datas) > 0:
            feed_dict = {self.input_images: raw_image_datas}
            predict_values, logit_values = self.sess.run(
                self.fetches, feed_dict)
            results.extend(normalize(predict_values))

        del raw_image_datas

        return results
