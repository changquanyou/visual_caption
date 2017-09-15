# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

BATCH_SIZE = 10  # default batch size
BASE_DATA_DIR = "/home/liuxiaoming/data/"
# BASE_DATA_DIR = "C:/Users/tsf/Desktop/gitdata/data/"
MODULE_NAME = "ai_challenge"  # default model name


class BaseDataConfig(object):
    def __init__(self, model_name=MODULE_NAME):
        self.base_data_dir = BASE_DATA_DIR
        self.batch_size = BATCH_SIZE

        self.model_name = model_name
        self.model_data_dir = os.path.join(self.base_data_dir,
                                           self.model_name)
