# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import time

import tensorflow as tf

from visual_caption.base.config.base_config import BaseConfig

max_grad_norm = 5
max_max_epoch = 100

# state learning rate information and configuration

learning_start_decay_step = 20000
learning_rate = 1.0e-0
learning_rate_decay = 0.96
learning_rate_step = 10000

dropout_keep_prob = 0.5
initializer_scale = 0.08
early_stopping = 100

display_and_summary_step = 10
valid_step = 10000

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用


class BaseModelConfig(BaseConfig):
    """
    base configuration for all models
    """

    def __init__(self, data_config, model_name):
        super(BaseModelConfig, self).__init__()
        self.mode = 'train'  # default mode is train

        # hyper params from data_config
        self.data_config = data_config
        self.model_name = model_name
        self.model_dir = os.path.join(self.project_dir, self.model_name)
        self.batch_size = self.data_config.reader_batch_size

        # default data type
        self.data_type = tf.float32
        self.initializer_scale = initializer_scale

        # parameters for monitoring the running of model
        self.display_and_summary_step = display_and_summary_step

        self.log_dir = os.path.join(self.model_dir, "log")

        localtime = time.localtime()
        timeString = time.strftime("%Y_%m_%d_%H_%M_%S", localtime)
        self.log_dir = os.path.join(self.log_dir, timeString)

        self.log_train_dir = os.path.join(self.log_dir, 'train')
        self.log_validation_dir = os.path.join(self.log_dir, 'validation')
        self.log_test_dir = os.path.join(self.log_dir, 'test')

        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoint")

        # parameters for model training
        self.dropout_keep_prob = dropout_keep_prob
        self.learning_rate = learning_rate  #
        self.start_decay_step = learning_start_decay_step
        self.decay_steps = learning_rate_step
        self.decay_rate = learning_rate_decay
        self.max_grad_norm = max_grad_norm
        self.max_max_epoch = max_max_epoch  # max

        self.valid_step = valid_step  # valid step i

        self.early_stopping = early_stopping

        # modules are subsets of the model
        self.module_name = ""
        self.module_dir = os.path.join(self.model_dir, self.module_name)

        # number of GPUs
        self.num_gpus = 1

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定80%的gpu显存
        config.gpu_options.allow_growth = True  # 程序按需申请内存
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess_config = config
