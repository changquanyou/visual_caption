# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import tensorflow as tf

# BATCH_SIZE = None
max_grad_norm = 5
max_max_epoch = 100



# Learning rate for the initial phase of training.
learning_initial_rate = 2.0
learning_rate_decay_factor = 0.5
learning_num_epochs_per_decay = 8.0


dropout_keep_prob = 0.5
initializer_scale = 0.08
early_stopping = 100


#base_dir = "/home/liuxiaoming/data/visual_caption/"
base_dir = "/home/liuxiaoming/data/visual_caption/"


class BaseConfig(object):
    def __init__(self, model_name):
        self.base_dir = base_dir
        self.model_name = model_name

        self.dropout_keep_prob = dropout_keep_prob

        self.learning_initial_rate = learning_initial_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.learning_num_epochs_per_decay = learning_num_epochs_per_decay

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        self.initializer_scale = initializer_scale
        # self.batch_size = BATCH_SIZE
        self.data_type = tf.float32

        self.early_stopping = early_stopping
        self.max_grad_norm = max_grad_norm
        self.max_max_epoch = max_max_epoch  # max 。

        self.module_dir = os.path.join(base_dir, self.model_name)

        self.data_dir = os.path.join(self.module_dir, "data")
        self.model_dir = os.path.join(self.module_dir, "model")
        self.log_dir = os.path.join(self.module_dir, "log")
        self.checkpoint_dir = os.path.join(self.module_dir, "checkpoint")

        self.gpu_options = (tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
        self.sess_config = tf.ConfigProto(gpu_options=self.gpu_options,
                                          allow_soft_placement=True,
                                          log_device_placement=False)
