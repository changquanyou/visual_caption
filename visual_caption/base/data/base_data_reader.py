# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from abc import ABCMeta, abstractmethod


class BaseDataReader(object):
    """
        Base Abstract Class for Data Reader of TFRecords
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_config):
        self.data_config = data_config

    def load_data_generator(self, mode):
        if mode == "train":
            data_generator = self.load_train_data()
        elif mode == "test":
            data_generator = self.load_test_data()
        elif mode == "infer":
            data_generator = self.load_infer_data()
        return data_generator

    @abstractmethod
    def load_train_data(self):
        raise NotImplementedError()

    @abstractmethod
    def load_test_data(self):
        raise NotImplementedError()

    @abstractmethod
    def load_validation_data(self):
        raise NotImplementedError()
