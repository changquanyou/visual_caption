# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

from tf_visgen.base.data.base_data_config import BaseDataConfig


class TextGenDataConfig(BaseDataConfig):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)
        self.toke2vec_file = os.path.join(self.base_data_dir, "word2vec/GoogleNews-vectors-negative300.bin")

    pass
