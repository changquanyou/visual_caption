# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import numpy as np
import tensorflow as tf
from scipy.io import savemat

from tf_visgen.visgen.config import VisgenDataConfig
from tf_visgen.visgen.data.data_loader import DataLoader
from tf_visgen.visgen.feature.vgg19 import Vgg19

vgg_data_dir = "/home/liuxiaoming/data/vgg"

config = VisgenDataConfig()


class FeatureManager(object):
    def __init__(self, sess, vgg_model):
        self.sess = sess
        self.vgg_model = vgg_model
        # self.batch_size = batch_size
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg_model.build(self.images)

    def get_vgg_feature(self, image_batch):
        feed_dict = {self.images: image_batch}
        fc7 = self.sess.run(self.vgg_model.fc7, feed_dict=feed_dict)
        print("batch_region_image_embedding.shape={}".format(fc7.shape))
        return fc7


class TextFeatureManager(object):
    def __init__(self):
        pass


def generate_image_vgg(file_path):
    """
    extracting batch images features and save to specified mat file
    :param file_path:mat file to save
    :return:
    """
    vgg19_npy_file = os.path.join(vgg_data_dir, "vgg19.npy")
    vgg_model = Vgg19(vgg19_npy_file)

    loader = DataLoader(config)
    image_gen = loader.load_images()

    with tf.Session() as sess:
        feature_manager = FeatureManager(sess, vgg_model)
        num = 0
        feature_list = []
        # with open(file_path, mode='ab') as feature_file:  # open mat file with append and binary mode
        for image_id_batch, image_batch in image_gen:
            # get batch image feature data with shape [batch,4096]
            feature_batch = feature_manager.get_vgg_feature(image_batch)
            # convert batch image ids to array and expend it to 2-D array with shape [batch,1]
            ids = np.expand_dims(np.asarray(image_id_batch), axis=1)
            # concatenate ids and feature_batch to [batch, 4097]
            batch = np.concatenate((ids, feature_batch), axis=1)
            feature_list.extend(batch.astype(np.float32))
            #feature_array=np.append(feature_array, batch.astype(np.float32))
            # save batch [image_id,image_feature] to mat variable with named images
            # savemat(feature_file, mdict={"images": batch.astype(np.float32)})
            if num >= 1000:
                print("number of features is {}".format(len(feature_list)))
                # break

            num += len(feature_batch)
            print("num={}".format(num))
        savemat(file_path, mdict={"image_feature": np.asarray(feature_list)})


def main():
    generate_image_vgg(config.image_feature_file)
    pass


if __name__ == '__main__':
    main()
