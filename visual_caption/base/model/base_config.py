# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import time

import tensorflow as tf
import os
from pathlib import Path


# BATCH_SIZE = None
max_grad_norm = 5
max_max_epoch = 100

# Learning rate for the initial phase of training.

dropout_keep_prob = 0.6
initializer_scale = 0.08
early_stopping = 100

home = str(Path.home())

BASE_DIR = os.path.join(home, 'workspace')  # base running dir for all models
# PROJECT_DIR  = os.path.join()


class BaseConfig(object):
    def __init__(self, model_name, mode):
        self.model_name = model_name
        self.mode = mode  # train,test or validation

        self.base_dir = BASE_DIR # base dir for model
        self.model_dir = os.path.join(self.base_dir, self.model_name) # for
        # self.module_dir = os.path.join(self.base_dir, self.model_name)

        # number of GPUs
        self.num_gpus = 1

        # data type and drop_keep_ratre
        self.data_type = tf.float32  # data type
        self.dropout_keep_prob = dropout_keep_prob
        self.initializer_scale = initializer_scale

        # leaning rate for training
        self.learning_rate_min = 0.01,  # min learning rate.
        self.learning_initial_rate = 0.2
        self.learning_rate_decay_factor = 0.5
        self.learning_decay_steps = 10000
        self.learning_decay_rate = 0.96

        self.early_stopping = early_stopping

        # for
        self.max_grad_norm = max_grad_norm
        self.max_max_epoch = max_max_epoch  # max 。
        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # for log and checkpoint
        self.log_dir = os.path.join(self.model_dir, "log")
        self.log_dir = os.path.join(self.log_dir, str(time.time()))
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoint")

        # for session and gpu
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu 90%的显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess_config = config
