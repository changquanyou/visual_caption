# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging
import os
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from visual_caption.utils.decorator_utils import define_scope, timeit


class BaseModel(object):
    """
        Base Abstraction Class for with Tensorflow framework
    """
    __metaclass__ = ABCMeta

    def __init__(self, config, data_reader):

        self.config = config
        self.data_reader = data_reader
        self.global_step = tf.Variable(initial_value=0, trainable=False, name="global_step",
                                        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.summary_writer = tf.summary.FileWriter(logdir=self.config.log_dir)
        self._build_model()

    def _next_device(self):
        """Round robin the gpu device. (Reserve last gpu for expensive op)."""
        if self._num_gpus == 0:
            return ''
        dev = '/gpu:%d' % self._cur_gpu
        if self._num_gpus > 1:
            self._cur_gpu = (self._cur_gpu + 1) % (self._num_gpus - 1)
        return dev

    def _get_gpu(self, gpu_id):
        if self.config.num_gpus <= 0 or gpu_id >= self.config.num_gpus:
            self.logger.error('error number of GPUs')
            return ''
        return '/gpu:%d' % gpu_id

    @timeit
    def _build_model(self):
        self._get_logger()
        self._build_embeddings()
        self._build_inputs()
        self._build_network()
        self._build_loss()
        self._build_optimizer()
        self._build_train_op()
        self._build_summaries()

    @timeit
    @define_scope(scope_name='embeddings')
    def _build_embeddings(self):
        pass

    @timeit
    @define_scope(scope_name='inputs')
    @abstractmethod
    def _build_inputs(self):
        """
         define placeholders and variables for inputs
        :return:
        """
        raise NotImplementedError()

    @timeit
    @define_scope(scope_name='graph')
    @abstractmethod
    def _build_graph(self):
        # build computation graph
        raise NotImplementedError()

    @timeit
    @define_scope(scope_name='losses')
    @abstractmethod
    def _build_losses(self):
        # define loss
        raise NotImplementedError()

    @abstractmethod
    def _build_fetches(self):
        # define default fetches for run_epoch
        raise NotImplementedError()

    @timeit
    @define_scope(scope_name='optimizer')
    def _build_optimizer(self):
        # num_examples_per_epoch = self._data_reader.num_examples_per_epoch
        # batch_size = self._data_reader.data_config.batch_size
        # num_batches_per_epoch = (num_examples_per_epoch / batch_size)
        # decay_steps = int(num_batches_per_epoch *
        #                   self.config.learning_decay_steps)

        self._learning_rate = tf.maximum(
            self.config.learning_rate_min,  # min learning rate.
            tf.train.exponential_decay(learning_rate=self.config.learning_initial_rate,
                                       global_step=self.global_step,
                                       decay_steps=self.config.learning_decay_steps,
                                       decay_rate=self.config.learning_decay_rate))

        # learning_rate = self._learning_rate
        learning_rate = self.config.learning_initial_rate
        tf.summary.scalar('learning_rate', learning_rate)
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    @timeit
    @define_scope(scope_name='train_op')
    def _build_train_op(self):
        """
        Set up the training ops.
        """
        trainables = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainables)

        with tf.device(self._get_gpu(self.config.num_gpus - 1)):
            clipped_gradients, global_norm = tf.clip_by_global_norm(t_list=gradients,
                                                                    clip_norm=self.config.max_grad_norm,
                                                                    name='clip_by_global_norm')
            tf.summary.scalar('global_norm', global_norm)

        grads_and_vars = zip(clipped_gradients, trainables)
        self.train_op = self._optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                                         global_step=self.global_step,
                                                         name='train_step')

    @timeit
    @define_scope(scope_name='summaries')
    def _build_summaries(self):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        self.summary_merged = tf.summary.merge_all()
        # adding embeddings into projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'token_embeddings'
        meta_file = os.path.join(self.data_reader.data_config.embedding_dir, "metadata.tsv")
        embed.metadata_path = meta_file
        projector.visualize_embeddings(self.summary_writer, config)

    @timeit
    def save_model(self, sess, global_step):
        model_name = self.config.model_name
        checkpoint_dir = self.config.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        self.logger.info("save model {} at step {}".format(model_name, global_step))

    @timeit
    def restore(self, sess, checkpoint=None):
        self._logger(" [*] Reading checkpoint...")
        model_name = self.config.model_name
        if checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if checkpoint:
            self._logger("restoring model parameters from {}".format(checkpoint.model_checkpoint_path))
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith(model_name)]
            self._saver.restore(sess, checkpoint)
            self._logger("model {} restored".format(model_name))
            return True
        else:
            return False

    @timeit
    def _get_logger(self):
        logger = logging.getLogger('tensorflow')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
        self._logger = logger



    pass
