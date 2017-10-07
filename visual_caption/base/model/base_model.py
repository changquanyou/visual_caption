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

    def __init__(self, config, data_reader):

        self.config = config

        self._data_reader = data_reader

        self._initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        self._summary_writer = tf.summary.FileWriter(logdir=self.config.log_dir)

        self._summary_list = []

        self._build_model()

    def _build_model(self):
        print("...building model...")
        self._get_logger()
        self._setup_global_step()

        self._build_inputs()
        self._build_network()
        self._build_loss()
        self._build_optimizer()
        self._build_train_op()
        self._build_summaries()
        self._build_fetches()
        print("...building model finished...")

    @abstractmethod
    def _build_inputs(self):
        # define inputs variables
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
    def _build_fetches(self):
        # define default fetches for run_epoch
        raise NotImplementedError()

    def _build_optimizer(self):
        print("......building optimizer begin......")
        with tf.name_scope("optimizer"):
            self._optimizer = tf.train.AdamOptimizer()
            # tf.summary.scalar('learning_rate', self.config.learning_rate)
        print("......building optimizer end......")

    def _build_train_op(self):
        print("......building train_op begin......")

        # num_examples_per_epoch = self._data_reader.num_examples_per_epoch
        num_examples_per_epoch = 10000

        learning_initial_rate = tf.constant(self.config.learning_initial_rate)
        batch_size = self._data_reader.data_config.batch_size

        num_batches_per_epoch = (num_examples_per_epoch / batch_size)

        decay_steps = int(num_batches_per_epoch *
                          self.config.learning_num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_initial_rate, global_step):
            return tf.train.exponential_decay(
                learning_initial_rate,
                global_step,
                decay_steps=decay_steps,
                decay_rate=self.config.learning_rate_decay_factor,
                staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

        with tf.name_scope("train_op") as scope_name:
            trainables = tf.trainable_variables()
            gradients = tf.gradients(self._cost, trainables)

            # # take gradients of cost w.r.t all trainable variables
            # clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)

            clipped_gradients = self.config.clip_gradients

            # Set up the training ops.
            self._train_op = tf.contrib.layers.optimize_loss(
                loss=self._cost,
                global_step=self._global_step,
                learning_rate=self.config.learning_initial_rate,
                optimizer=self._optimizer,
                clip_gradients=clipped_gradients,
                learning_rate_decay_fn=learning_rate_decay_fn
            )
            # tf.summary.scalar('train_op', self._train_op)
        print("......building train_op end......")

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
        # Create a coordinator and run all QueueRunner objects
        print("......begin " + mode + " epoch {}.....".format(num_epoch))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        total_loss = 0.0
        try:
            for batch_index in range(5000):
                start = time.time()
                global_step = tf.train.global_step(sess, self._global_step)
                _, batch_loss, batch_summary = sess.run(self._fetches)
                total_loss = total_loss + batch_loss
                self.logger.info('global_step={}, batch_loss={}'.format(global_step, batch_loss))
                self._summary_writer.add_summary(batch_summary, global_step=global_step)
                if mode == "train":
                    if global_step % 500 == 0 and global_step > 0:
                        self.logger.info("epoch={}, batch_index={}, global_step={}, batch_loss={},batch_time={}".
                                         format(num_epoch, batch_index, global_step, batch_loss, time.time() - start))

        except Exception as e:
            print(e)
            coord.request_stop(e)
        finally:
            coord.request_stop()  # Stop the threads
            coord.join(threads)  # Wait for threads to stop

        print("......end " + mode + " epoch {}.....".format(num_epoch))
        return global_step, total_loss

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

    def _setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self._global_step = global_step

    def run_train(self):
        print("......begin training......")
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
                # 更新并保存最优 epoch
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
        print("......end training.....")

    @abstractmethod
    def _run_test(self, sess, global_step):
        raise NotImplementedError()

    def _run_validate(self, sess):

        pass
