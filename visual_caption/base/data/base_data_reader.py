# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from abc import ABCMeta, abstractmethod


class BaseDataReader(object):
    """
    Base Abstract Data Reader Class of TFRecords
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_config):
        self.data_config = data_config

    def build_data_inputs(self):
        if self.data_config.mode == "train":
            self.data_inputs = self._build_data_inputs(data_dir=self.data_config.train_data_dir)
        elif self.data_config.mode == "test":
            self.data_inputs = self._build_data_inputs(data_dir=self.data_config.test_data_dir)
        elif self.data_config.mode == "validation":
            self.data_inputs = self._build_data_inputs(data_dir=self.data_config.validation_data_dir)
        return self.data_inputs

    @abstractmethod
    def _build_data_inputs(self, data_dir):
        raise NotImplementedError()
