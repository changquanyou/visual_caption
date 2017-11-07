# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from  visual_caption.base.model.base_model_config import BaseConfig


class TextGenConfig(BaseConfig):
    def __init__(self, model_name):
        super().__init__(model_name=model_name)
        self.hidden_layer_num = 2
        self.hidden_neural_num = 512
        self.train_embeddings = False
        self.vocab_size=4350
        self.embedding_size=300
        self._input_phrases=0

    pass
