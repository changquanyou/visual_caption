# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from functools import wraps

import tensorflow as tf


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        print('......   begin  {}   ......'.format(f.__name__))
        result = f(*args, **kw)
        te = time.time()
        print('......   finish {}, took: {} sec   ......'.format(f.__name__, te - ts))
        return result

    return timed


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope_name=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope_name or function.__name__

    @wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


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

    @timeit
    def _build_model(self):
        self._get_logger()
        self._setup_global_step()
        self._build_inputs()
        self._build_network()
        self._build_loss()
        self._build_optimizer()
        self._build_train_op()
        self._build_summaries()
        self._build_fetches()

    @timeit
    @define_scope(scope_name='inputs')
    @abstractmethod
    def _build_inputs(self):
        # define inputs variables
        raise NotImplementedError()

    @timeit
    @define_scope(scope_name='network')
    @abstractmethod
    def _build_network(self):
        # define deep network of computation graph
        raise NotImplementedError()

    @timeit
    @define_scope(scope_name='losses')
    @abstractmethod
    def _build_loss(self):
        # define loss
        raise NotImplementedError()

    @abstractmethod
    def _build_fetches(self):
        # define default fetches for run_epoch
        raise NotImplementedError()

    @timeit
    @define_scope(scope_name='optimizer')
    def _build_optimizer(self):
        self._optimizer = tf.train.AdamOptimizer()

    @timeit
    @define_scope(scope_name='train_op')
    def _build_train_op(self):

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

    @timeit
    @define_scope(scope_name='summaries')
    def _build_summaries(self):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        for (summary_key, summary_value) in self._summary_list:
            tf.summary.scalar(summary_key, summary_value)
            tf.summary.histogram(summary_key, summary_value)
        self._merged = tf.summary.merge_all()

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

    @timeit
    def _save_model(self, sess, global_step):
        model_name = self.config.model_name
        checkpoint_dir = self.config.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        self.logger.info("save model {} at step {}".format(model_name, global_step))

    @timeit
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

    @timeit
    def _get_logger(self):
        logger = logging.getLogger("logger")
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
        self.logger = logger

    @timeit
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
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            train_fetches = self._build_fetches()
            try:
                start = time.time()
                while not coord.should_stop():
                    global_step = tf.train.global_step(sess, self._global_step)
                    _, batch_loss, batch_summary = sess.run(fetches=train_fetches)
                    self._summary_writer.add_summary(batch_summary, global_step=global_step)
            except tf.errors.OutOfRangeError:
                print("Done training after reading all data")
            except Exception as exception:
                print(exception)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
            finally:
                # finalise
                coord.request_stop()  # Stop the threads
                coord.join(threads)  # Wait for threads to stop

            self._summary_writer.close()
        print("......end training.....")

    @abstractmethod
    def _run_test(self, sess, global_step):
        raise NotImplementedError()

    @abstractmethod
    def _run_validation(self, sess, global_step):
        raise NotImplementedError()

    @abstractmethod
    def _run_inference(self):
        raise NotImplementedError()
    pass
