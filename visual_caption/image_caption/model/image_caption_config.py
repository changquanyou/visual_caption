# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from visual_caption.base.model.base_config import BaseConfig


class ImageCaptionConfig(BaseConfig):

    def __init__(self, model_name):
        super().__init__(model_name=model_name)
    pass
