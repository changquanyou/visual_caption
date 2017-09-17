

# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf

class BaseDataPrepare(object):
    """
    Base Class for Data Prepare

    aim to fetch raw data and convert them into TFRecord or HDF5 Format

    """

    def _int64_feature(self,value):
        """Wrapper for inserting an int64 Feature into a SequenceExample proto,
        e.g, An integer label.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self,value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto,
        e.g, an image in byte
        """
        # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature_list(self,values):
        """Wrapper for inserting an int64 FeatureList into a SequenceExample proto,
        e.g, sentence in list of ints
        """
        return tf.train.FeatureList(feature=[self._int64_feature(v) for v in values])

    def _bytes_feature_list(self,values):
        """Wrapper for inserting a bytes FeatureList into a SequenceExample proto,
        e.g, sentence in list of bytes
        """
        return tf.train.FeatureList(feature=[self._bytes_feature(v) for v in values])