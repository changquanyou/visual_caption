# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf

from visual_caption.base.model.base_model import BaseModel


class ImageCaptionModel(BaseModel):
    def __init__(self, config, data_loader):
        super().__init__(config=config, data_loader=data_loader)

    def _build_placeholder(self):
        # input images batch
        self.inputs = tf.placeholder(dtype=self.config.data_type,
                                     shape=[self.config.batch_size],
                                     name="inputs")
        # target caption batch
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[self.config.batch_size],
                                      name="targets")

    def _build_CNN(self):
        pass

    def _build_RNN(self):
        pass

    def _build_network(self):
        # CNN + RNN
        pass

    def _build_feed_and_fetch(self, batch_data):
        (image_batch, caption_batch) = batch_data
        feed_dict = {
            self.inputs: image_batch,
            self.targets: caption_batch
        }
        fetches = [self._train_op, self._cost, self._merged]
        pass

    def _build_loss(self):
        print("......building loss......")
        # Compute logits and weights
        hidden_size = self.config.hidden_neural_num
        vocab_size = self._data_loader.vocab_size
        data_type = self.config.data_type
        batch_size = self.config.batch_size

        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", [hidden_size * 2, vocab_size], dtype=data_type)
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type)

        with tf.variable_scope("logits"):
            logits = tf.matmul(self._outputs, softmax_w) + softmax_b  # logits shape[time_step, target_num]

        # Computing losses.
        with tf.variable_scope("loss"):
            # adding extra statistics to monitor
            targets = self._targets
            correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(targets, [-1]))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]), logits=logits)
            self._cost = tf.reduce_mean(loss)  # loss
            tf.summary.scalar("accuracy", self._accuracy)
            tf.summary.scalar("loss", self._cost)
        print("......building loss finished......")
        pass
