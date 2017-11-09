# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from  visual_caption.base.model.base_model_config import BaseConfig


class TextGenConfig(BaseConfig):
    def __init__(self, model_name):
        super(TextGenConfig, self).__init__()
        self.num_layers = 2
        self.num_units = 512
        self.train_embeddings = False
        self.model_name = model_name

    pass
