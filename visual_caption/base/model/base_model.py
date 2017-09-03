# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging
import os
import time
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class BaseModel(object):
    """
        Base Abstraction Class for with Tensorflow framework
    """
    __metaclass__ = ABCMeta

    def __init__(self, config, data_loader):

        self.config = config

        self._data_loader = data_loader

        self._global_step = tf.Variable(0, trainable=False, name="global_step")

        self._initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        self._summary_writer = tf.summary.FileWriter(logdir=self.config.log_dir)

        self._summary_list = []

        self._build_model()

    def _build_model(self):
        print("...building model...")
        self._get_logger()
        self._build_placeholder()
        self._build_network()
        self._build_loss()
        self._build_optimizer()
        self._build_train_op()
        self._build_summaries()
        print("...building model finished...")

    @abstractmethod
    def _build_placeholder(self):
        # define placeholders and variables
        raise NotImplementedError()

    @abstractmethod
    def _build_network(self):
        # define deep network of computation graph
        raise NotImplementedError()

    @abstractmethod
    def _build_loss(self):
        # define loss
        raise NotImplementedError()

    @abstractmethod
    def _build_feed_and_fetch(self):
        # define default feed_dict and fetches for run_epoch
        raise NotImplementedError()

    def _build_optimizer(self):
        print("......building optimizer......")
        with tf.name_scope("optimizer"):
            # self.config.learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate,
            #                                                        global_step=self._global_step,
            #                                                        decay_steps=self.config.decay_steps,
            #                                                        decay_rate=self.config.decay_rate,
            #                                                        staircase=True)

            self._optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            tf.summary.scalar('learning_rate', self.config.learning_rate)
        print("......building optimizer......")

    def _build_train_op(self):
        print("......building train_op......")
        with tf.name_scope("train_op") as scope_name:
            trainables = tf.trainable_variables()
            gradients = tf.gradients(self._cost, trainables)
            # take gradients of cost w.r.t all trainable variables
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
            self._train_op = self._optimizer.apply_gradients(zip(gradients, trainables),
                                                             global_step=self._global_step)
            # tf.summary.scalar('train_op', self._train_op)
        print("......building train_op......")

    def _build_summaries(self):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        print("......building summary begin......")
        with tf.name_scope("summaries"):
            for (summary_key, summary_value) in self._summary_list:
                tf.summary.scalar(summary_key, summary_value)
                tf.summary.histogram(summary_key, summary_value)
            # merge them all
            self._merged = tf.summary.merge_all()
        print("......building summary end......")

    def _run_epoch(self, sess, num_epoch, mode):
        """
        running epoch
        :param num_epoch: number of epoch
        :return:
        """
        data_generator = self._data_loader.load_data_generator(mode=mode)

        for batch_order, batch_data in enumerate(data_generator):
            start = time.time()
            global_step = tf.train.global_step(sess, self._global_step)

            feed_dict, fetches = self._get_feed_and_fetch(batch_data=batch_data)

            _, batch_loss, batch_summary = sess.run(fetches, feed_dict)

            # print("batch = {}, batch_loss = {}, global_step = {}".format(batch_order, batch_loss, global_step))

            self._summary_writer.add_summary(batch_summary, global_step=global_step)

            if mode == "train":
                if global_step % 500 == 0 and global_step > 0:
                    # self._save_model(sess=sess, global_step=global_step)
                    self.logger.info("epoch={}, batch={}, global_step={}, batch_loss={},batch_time={}".
                                     format(num_epoch, batch_order, global_step, batch_loss, time.time() - start))
                    # elif mode == "test":
                    #     self._run_test(global_step=global_step)
                    # else:
                    #     print("")

        return global_step, batch_loss

    def _save_model(self, sess, global_step):
        model_name = self.config.model_name
        checkpoint_dir = self.config.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        self.logger.info("save model {} at step {}".format(model_name, global_step))

    def _restore_model(self, checkpoint):
        print(" [*] Reading checkpoint...")
        model_name = self.config.model_name
        if checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)

        if checkpoint:
            print("restoring model parameters from {}".format(checkpoint.model_checkpoint_path))
            all_vars = tf.global_variables()
            model_vars = [k for k in all_vars if k.name.startswith(model_name)]
            self._saver.restore(self.sess, checkpoint)
            print("model {} restored".format(model_name))
            return True
        else:
            return False

    def _get_logger(self):
        logger = logging.getLogger("logger")
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
        self.logger = logger

    def run_train(self):
        with tf.Session(config=self.config.sess_config) as sess:
            self._summary_writer.add_graph(sess.graph)
            epoch_size = self.config.max_max_epoch
            checkpoint_dir = self.config.checkpoint_dir
            saver = tf.train.Saver()

            # CheckPoint State
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
            if checkpoint:
                self.logger.info("Loading model parameters from {}".format(checkpoint.model_checkpoint_path))
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            else:
                self.logger.info("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())

            # 最优解
            best_val_acc = 0.0
            best_val_epoch = 0

            for epoch in range(epoch_size):
                start = time.time()

                global_step, epoch_loss = self._run_epoch(num_epoch=epoch, sess=sess, mode="train")

                valid_acc = self._run_test(sess=sess, global_step=global_step)

                tf.summary.scalar("valid_accuracy", valid_acc)

                self.logger.info(
                    "epoch={},global_step={},epoch_loss={},valid_acc={}".format(epoch, global_step, epoch_loss,
                                                                                valid_acc))
                # 更新最优迭代步骤
                if valid_acc > best_val_acc:
                    best_val_acc = valid_acc
                    best_val_epoch = epoch
                    self._save_model(sess=sess, global_step=global_step)

                print('best valid_acc is {} in epoch: {}'.format(best_val_acc, best_val_epoch))
                self.logger.info('Total time for running epoch {} is : {}'.format(epoch, time.time() - start))
                # early_stopping
                if epoch - best_val_epoch > self.config.early_stopping:
                    break

            self._summary_writer.close()

    @abstractmethod
    def _run_test(self, sess, global_step):
        raise NotImplementedError()
