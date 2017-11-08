# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class BaseDataReader(object):
    __metaclass__ = ABCMeta

    def __init__(self, data_config):
        self._data_config = data_config

        # default batch_size from data reader config
        self._batch_size = self._data_config.reader_batch_size
        self._build_context_and_feature()
        self.data_iterator = self.get_data_iterator()
        pass

    def get_data_iterator(self):
        """
        get a data iterator for all dataset including train,valid and test
        :return:
        """
        dataset = self._get_dataset(self._data_config.train_data_dir)
        data_iterator = tf.data.Iterator.from_structure(
            output_types=dataset.output_types,
            output_shapes=dataset.output_shapes
        )
        return data_iterator

    def get_next_batch(self, batch_size=None):
        if batch_size:
            self._batch_size = batch_size
        next_batch = self.data_iterator.get_next()
        return next_batch

    def get_train_init_op(self):
        _train_dataset = self._get_dataset(data_dir=self._data_config.train_data_dir)
        initializer = self.data_iterator.make_initializer(_train_dataset)
        return initializer

    def get_valid_init_op(self):
        _valid_dataset = self._get_dataset(data_dir=self._data_config.valid_data_dir)
        initializer = self.data_iterator.make_initializer(_valid_dataset)
        return initializer

    def get_test_init_op(self):
        _test_dataset = self._get_dataset(data_dir=self._data_config.test_data_dir)
        initializer = self.data_iterator.make_initializer(_test_dataset)
        return initializer

    def _get_dataset(self, data_dir):
        """
        get tf.data.TFRecordDataset from give data_dir and mapping them into dataset
        :param data_dir:
        :return:
        """
        filenames = os.listdir(data_dir)
        data_files = []
        for filename in filenames:
            data_file = os.path.join(data_dir, filename)
            data_files.append(data_file)
        dataset = tf.data.TFRecordDataset(data_files)
        # parsing tf_record
        dataset = dataset.map(self._parse_tf_example)
        # mapping dataset
        dataset = self._mapping_dataset(dataset)  # mapping to target format
        return dataset

    @abstractmethod
    def _mapping_dataset(self, dataset):
        """mapping data to necessary format  """
        raise NotImplementedError()
        pass

    @abstractmethod
    def _build_context_and_feature(self):
        """used by tfrecord parsing"""
        raise NotImplementedError()
        pass

    @abstractmethod
    def _parse_tf_example(self):
        """parsing tfrecord parsing"""
        raise NotImplementedError()
        pass
