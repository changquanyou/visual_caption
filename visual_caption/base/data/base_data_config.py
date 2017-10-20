# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from pathlib import Path

home = str(Path.home())

BASE_DATA_DIR = os.path.join(home, 'data/ai_challenge')  # base data dir

MODE = 'train'  # default mode, mode should be one of {'train','test','validation'}
BATCH_SIZE = 20  # default batch_size


class BaseDataConfig(object):
    def __init__(self, model_name, mode=MODE):
        self.mode = mode
        self.model_name = model_name
        self.batch_size = BATCH_SIZE
        self.base_data_dir = BASE_DATA_DIR
        self.model_data_dir = os.path.join(self.base_data_dir,
                                           self.model_name)
