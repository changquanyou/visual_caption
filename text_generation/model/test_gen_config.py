# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from  tf_visgen.base.model.base_config import BaseConfig


class TextGenConfig(BaseConfig):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)
        self.hidden_layer_num = 2
        self.hidden_neural_num = 512
        self.train_embeddings = False

    pass
