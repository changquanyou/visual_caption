# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from abc import ABCMeta, abstractmethod


class BaseDataLoader(object):
    """
        Base Abstraction Data Loader Class for raw data
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_config):
        self.data_config = data_config