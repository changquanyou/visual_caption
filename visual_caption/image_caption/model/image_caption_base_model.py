# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from visual_caption.base.model.base_model import BaseModel
from visual_caption.utils.decorator_utils import timeit, define_scope


class ImageCaptionBaseModel(BaseModel):
    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionBaseModel, self).__init__(
            model_config, data_reader, mode)
        self.loss = None

    @timeit
    @define_scope(scope_name='optimizer')
    def _build_optimizer(self):
        config = self.model_config
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            tf.summary.scalar('learning_rate', config.learning_rate)

    @timeit
    @define_scope(scope_name='gradients')
    def _build_gradients(self):
        """Clipping gradients of a model."""
        # if self.mode is not ModeKeys.INFER:
        #     trainables = tf.trainable_variables()
        #     with tf.device(self._get_gpu(self.model_config.num_gpus - 1)):
        #         gradients = tf.gradients(self.loss, trainables)
        #         # clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        #         #     gradients, self.model_config.max_grad_norm)
        #         self._gradients = gradients
        #         # tf.summary.scalar("grad_norm", gradient_norm)
        #         tf.summary.scalar("clipped_gradient", tf.global_norm(gradients))

    @timeit
    @define_scope(scope_name='train_op')
    def _build_train_op(self):
        if self.mode == ModeKeys.TRAIN:
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
