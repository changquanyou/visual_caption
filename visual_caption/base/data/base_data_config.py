# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

from visual_caption.base.config.base_config import BaseConfig


class BaseDataConfig(BaseConfig):
    """
        set default parameters for data config
    """

    def __init__(self, model_name, mode='train'):
        super(BaseDataConfig, self).__init__()

        """set default model dirs"""

        self.builder_batch_size = self.batch_size
        self.reader_batch_size = self.batch_size * 3
        self.mode = mode

        """set default model data dirs"""
        self.model_name = model_name
        self.model_data_dir = os.path.join(self.project_data_dir, self.model_name)

        self.train_data_dir = os.path.join(self.model_data_dir, "train")
        self.test_data_dir = os.path.join(self.model_data_dir, "test")
        self.valid_data_dir = os.path.join(self.model_data_dir, "valid")

        """set default of model running"""
        self.num_epoches = 10

        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None
        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 10000
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_threads = 4

        # Number of threads for preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4
        self.output_buffer_size = 1000
        self.random_seed = 123
