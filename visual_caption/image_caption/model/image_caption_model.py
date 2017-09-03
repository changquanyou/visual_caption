# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from visual_caption.base.model.base_model import BaseModel


class ImageCaptionModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config=config, data_loader=data_loader)

    pass
