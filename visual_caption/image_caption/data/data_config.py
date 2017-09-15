# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

from visual_caption.base.data.base_data_config import BaseDataConfig


class ImageCaptionDataConfig(BaseDataConfig):
    def __init__(self):
        super().__init__()

        self.train_data_dir = os.path.join(self.model_data_dir,"ai_challenger_caption_train_20170902")
        self.train_json_data = os.path.join(self.train_data_dir,"caption_train_annotations_20170902.json")
        self.train_image_dir = os.path.join(self.train_data_dir,"caption_train_images_20170902")

        self.validation_data_dir = os.path.join(self.model_data_dir, "ai_challenger_caption_validation_20170910")
        self.validation_json_data = os.path.join(self.validation_data_dir, "caption_validation_annotations_20170910.json")
        self.validation_image_dir = os.path.join(self.validation_data_dir, "caption_validation_images_20170910")

    pass
