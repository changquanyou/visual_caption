# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from abc import ABCMeta, abstractmethod

from visual_caption.utils.decorator_utils import timeit


class BaseRunner(object):
    """
    Base Abstraction Class for Module Runner with Tensorflow framework
    """
    __metaclass__ = ABCMeta

    @timeit
    @abstractmethod
    def _internal_eval(self, model, sess):
        raise NotImplementedError()

    @timeit
    @abstractmethod
    def train(self):
        raise NotImplementedError()
        pass

    @timeit
    @abstractmethod
    def eval(self):
        raise NotImplementedError()
        pass

    @timeit
    @abstractmethod
    def infer(self):
        raise NotImplementedError()
        pass
