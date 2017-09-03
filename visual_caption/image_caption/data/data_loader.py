# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

from visual_caption.base.data.base_data_loader import BaseDataLoader


class ImageCaptionDataLoader(BaseDataLoader):

    def __init__(self, data_config):
       super().__init__(data_config=data_config)

    pass
