# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from pathlib import Path

import tensorflow as tf
from sklearn.preprocessing import normalize

from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.utils import image_utils
from visual_caption.utils.decorator_utils import timeit

slim = tf.contrib.slim
from slim.nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

home = str(Path.home())  # home dir
base_data_dir = os.path.join(home, 'data')
model_data_dir = os.path.join(base_data_dir, "tf/models")


class InceptionResnetV2FeatureExtractor(object):
    """
    inception_resnet_v2 feature extractor

        input batched image_paths
        output features for input image_paths


    """

    def __init__(self, sess=None):

        self.inception_resnet_v2_ckpt = os.path.join(
            model_data_dir, "inception_resnet_v2_2016_08_30.ckpt")
        self.input_images = tf.placeholder(shape=[None, 299, 299, 3],
                                           dtype=tf.float32, name='input_images')
        if sess is None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            sess_config = tf.ConfigProto(gpu_options=gpu_options,
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
            logits, end_points = inception_resnet_v2(scaled_input_tensor, is_training=False)
        saver = tf.train.Saver()
        checkpoint_file = self.inception_resnet_v2_ckpt
        saver.restore(self.sess, checkpoint_file)
        self.fetches = [end_points['PreLogitsFlatten'], logits]

    def get_features(self, images):
        raw_images = image_utils.load_images(images)
        feed_dict = {self.input_images: raw_images}
        predict_values, logit_values = self.sess.run(self.fetches,
                                                     feed_dict)
        results = normalize(predict_values)
        return results


def load_images():
    data_config = ImageCaptionDataConfig()
    batch_size = 101
    image_files = list()
    for file_path in Path(data_config.test_image_dir).glob('**/*'):
        image_files.append(file_path.absolute())
        if len(image_files) == batch_size:
            yield image_files
            image_files = list()
    if len(image_files) > 0:
        yield image_files


def get_img_id(img_path):
    arrays = str(img_path).split('/')
    return arrays[len(arrays) - 1].split('.')[0]


def main(_):
    feature_extractor = InceptionResnetV2FeatureExtractor()
    data_gen = load_images()
    for batch, batch_data in enumerate(data_gen):
        features = feature_extractor.get_features(images=batch_data)
        print("batch={:4d}, batch_size={:4d}".format(batch, len(batch_data)))
        for idx, image_path in enumerate(batch_data):
            print("\tidx={:4d}, image_id={:20}, feature_length={:4d}"
                  .format(idx, get_img_id(image_path), len(features[idx])))


if __name__ == '__main__':
    tf.app.run()
