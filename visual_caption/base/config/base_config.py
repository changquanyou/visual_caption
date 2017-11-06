# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from pathlib import Path

home = str(Path.home())  # home dir
project_name = 'ai_challenge'  # project name

BASE_WORK_DIR = os.path.join(home, 'workspace')  # base work space dir for all models running
BASE_DATA_DIR = os.path.join(home, 'data')  # base data dir for models

PROJECT_DATA_DIR = os.path.join(BASE_DATA_DIR, project_name)  # base data dir for project
PROJECT_DIR = os.path.join(BASE_WORK_DIR, project_name)  # base workspace dir for project
BATCH_SIZE = 20  # default batch size


class BaseConfig(object):
    def __init__(self):
        self.batch_size = BATCH_SIZE

        self.base_work_dir = BASE_WORK_DIR
        self.base_data_dir = BASE_DATA_DIR

        self.project_data_dir = PROJECT_DATA_DIR
        self.project_dir = PROJECT_DIR
