# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import tensorflow as tf
from scipy.misc import imresize

from visual_caption.image_caption.feature.vgg19 import Vgg19

vgg_data_dir = "/home/liuxiaoming/data/vgg"


class FeatureManager(object):
    def __init__(self, vgg_model_dir=vgg_data_dir, sess=None):
        vgg19_npy_file = os.path.join(vgg_model_dir, "vgg19.npy")
        self.vgg_model = Vgg19(vgg19_npy_file)
        self.shape = (224, 224, 3)

        # self.batch_size = batch_size
        self.input_images = tf.placeholder(tf.float32, [None, 224, 224, 3], name="input_images")

        if sess is None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                         allow_soft_placement=True,
                                         log_device_placement=False)
            self.sess = tf.Session(config=sess_config)
        else:
            self.sess = sess

        self.vgg_model.build(self.input_images)

    def get_vgg_feature(self, image_batch):
        image_rawdata_batch = []
        for img in image_batch:
            # print("img.shape={}".format(img.shape))
            resized_image = imresize(img, self.shape)
            image_rawdata_batch.append(resized_image)

        feed_dict = {self.input_images: image_rawdata_batch}
        fc7 = self.sess.run(self.vgg_model.fc7, feed_dict=feed_dict)
        print("shape={}".format(fc7.shape))
        # result = np.asarray(fc7)
        return fc7
