# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from visual_caption.base.model.base_model_config import BaseModelConfig


class ImageCaptionConfig(BaseModelConfig):
    def __init__(self, data_config, model_name):
        super(ImageCaptionConfig, self).__init__(
            data_config=data_config,
            model_name=model_name
        )

        self.residual = False
        self.forget_bias = 1.0
        self.num_residual_layers = 0

        # for hidden unit
        self.unit_type = "lstm"  # RNN Cell unit type : ‘lstm’,'gru' or 'nas'
        self.num_layers = 1  # number of hidden layers
        self.num_hidden_unit = 64  # number of neural units in each hidden layers

        self.length_max_output = 64

        self.train_embeddings = False
        self.pass_hidden_state = False  # whether passing hidden or not during decoding

        self.attention_mechanism = "normed_bahdanau"
        self.num_attention_unit = 100
        self.num_attention_layer = 20
        self.beam_width = 0
        pass

    pass
