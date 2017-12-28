# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging
import os
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from visual_caption.utils.decorator_utils import timeit, define_scope


class BaseModel(object):
    """
        Base Abstraction Class for with Tensorflow framework
    """

    __metaclass__ = ABCMeta

    def __init__(self, model_config, data_reader, mode):

        self.model_config = model_config
        self.data_reader = data_reader

        self.data_type = self.model_config.data_type
        self.mode = mode

        self.summary_writer = tf.summary.FileWriter(logdir=self.model_config.log_train_dir)
        self.summary_valid_writer = tf.summary.FileWriter(logdir=self.model_config.log_valid_dir)
        self.summary_test_writer = tf.summary.FileWriter(logdir=self.model_config.log_test_dir)

        # for model data pipeline
        self.batch_size = self.model_config.batch_size
        self.next_batch = self.data_reader.get_next_batch(batch_size=self.batch_size)
        self.initializer = tf.random_uniform_initializer(
            minval=-self.model_config.initializer_scale,
            maxval=self.model_config.initializer_scale)

        # To match the "Show Attend Tell" paper we initialize all variables with a
        # truncated_normal initializer.
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # build model
        self._build_model()

    @timeit
    def _build_model(self):
        """
        build model
        :return:
        """
        self._get_logger()
        self._build_global_step()
        self._build_inputs()
        self._build_embeddings()
        self._build_graph()
        self._build_loss()
        self._build_optimizer()
        self._build_gradients()
        self._build_train_op()
        self._build_summaries()
        # create a model saver to save or restore model
        self.model_server = tf.train.Saver()

    @timeit
    @define_scope(scope_name='global_step')
    def _build_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step_tensor = global_step

    @abstractmethod
    def _build_embeddings(self):
        raise NotImplementedError()

    @abstractmethod
    def _build_inputs(self):
        raise NotImplementedError()

    @timeit
    @abstractmethod
    def _build_graph(self):
        raise NotImplementedError()
        pass

    @timeit
    @abstractmethod
    def _build_loss(self):
        raise NotImplementedError()
        pass

    @timeit
    @define_scope(scope_name='optimizer')
    def _build_optimizer(self):
        config = self.model_config
        # Gradients and SGD update operation for training the model.
        # Arrange for the embedding vars to appear at the beginning.
        self.learning_rate = tf.constant(config.learning_rate)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.cond(
                self.global_step_tensor < config.start_decay_step,
                lambda: self.learning_rate,
                lambda: tf.train.exponential_decay(
                    learning_rate=self.learning_rate,
                    global_step=(self.global_step_tensor - config.start_decay_step),
                    decay_steps=config.decay_steps,
                    decay_rate=config.decay_rate,
                    staircase=True))
            tf.summary.scalar('learning_rate', self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    @timeit
    @define_scope(scope_name='gradients')
    def _build_gradients(self):
        """Clipping gradients of a model."""
        if self.mode is not ModeKeys.INFER:
            trainables = tf.trainable_variables()
            with tf.device(self._get_gpu(self.model_config.num_gpus - 1)):
                gradients = tf.gradients(self.loss, trainables)
                # clipped_gradients, gradient_norm = tf.clip_by_global_norm(
                #     gradients, self.model_config.max_grad_norm)
                self._gradients = gradients
                # tf.summary.scalar("grad_norm", gradient_norm)
                tf.summary.scalar("clipped_gradient", tf.global_norm(gradients))

    @timeit
    @define_scope(scope_name='train_op')
    def _build_train_op(self):
        if self.mode is not ModeKeys.INFER:
            trainables = tf.trainable_variables()
            grads_and_vars = zip(self._gradients, trainables)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                                           global_step=self.global_step_tensor,
                                                           name='train_step')

    @timeit
    @define_scope(scope_name='summaries')
    def _build_summaries(self):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)
        self.summary_merged = tf.summary.merge_all()

    def _next_device(self):
        """Round robin the gpu device. (Reserve last gpu for expensive op)."""
        if self.model_config.num_gpus == 0:
            return ''
        dev = '/gpu:%d' % self._cur_gpu
        if self.model_config.num_gpus > 1:
            self._cur_gpu = (self._cur_gpu + 1) % (self.model_config.num_gpus - 1)
        return dev

    @timeit
    def _get_logger(self):
        logger = logging.getLogger('tensorflow')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
        self.logger = logger

    def _get_gpu(self, gpu_id):
        if self.model_config.num_gpus <= 0 or gpu_id >= self.model_config.num_gpus:
            self.logger.error('error number of GPUs')
            return ''
        return '/gpu:%d' % gpu_id

    @timeit
    def save_model(self, sess, global_step):
        model_name = self.model_config.model_name
        checkpoint_dir = self.model_config.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_server.save(sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        self.logger.info("save model {} at step {}".format(model_name, global_step))

    @timeit
    def restore_model(self, sess, checkpoint_path=None):
        """
        :param sess:
        :param checkpoint_path: if checkpoint is None, restore from last checkpoint
        :return:
        """
        print(" [*] Reading checkpoint...")
        if checkpoint_path is None:
            checkpoint_path = tf.train.latest_checkpoint(self.model_config.checkpoint_dir)

        if checkpoint_path:
            print("restoring model parameters from {}".format(checkpoint_path))
            # all_vars = tf.global_variables()
            # model_vars = [k for k in all_vars if k.name.startswith(model_name)]
            self.model_server.restore(sess, checkpoint_path)
            print("model  restored".format())
            return True
        else:
            return False
