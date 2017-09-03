# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from abc import ABCMeta, abstractmethod


class BaseRunner(object):
    """
    Base Abstraction Class for Module Runner with Tensorflow framework
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        raise NotImplementedError()

    @abstractmethod
    def infer(self):
        raise NotImplementedError()
