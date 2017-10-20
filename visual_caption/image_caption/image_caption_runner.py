# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging
import sys
import time

import tensorflow as tf

from visual_caption.base.base_runner import BaseRunner
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_reader import ImageCaptionDataReader
from visual_caption.image_caption.model.image_caption_config import ImageCaptionConfig
from visual_caption.image_caption.model.image_caption_model import ImageCaptionModel
from visual_caption.utils.decorator_utils import timeit


class ImageCaptionRunner(BaseRunner):
    def __init__(self):
        self.model_name = 'image_caption'
        self.get_logger()
        pass

    def run_epoch(self, sess, model):
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        fetches = [model.accuracy, model.loss, model.logits, model.summary_merged, model.train_op]
        try:
            start = time.time()
            # batch_count = 0
            while not coord.should_stop():
                global_step = tf.train.global_step(sess, model.global_step)
                accuracy, loss, logits, summaries, _ = sess.run(fetches=fetches)
                model.summary_writer.add_summary(summaries, global_step=global_step)
                # batch_count += 1
                if global_step % 100 == 0 and global_step > 0:
                    last = time.time() - start
                    print('global_step={}, accuracy={}, loss={}, time={}'.format(global_step, accuracy,
                                                                                 loss, last))
                if model.config.mode == 'train':
                    if global_step % 10000 == 0 and global_step > 0:
                        model.save_model(global_step=global_step)
                        last = time.time() - start
                        print('global_step={}, loss={}, time={}'.format(global_step, loss, last))

        except tf.errors.OutOfRangeError:
            print("Done training after reading all data")
        except Exception as exception:
            print("Exception:{}".format(exception))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        finally:
            # finalise
            coord.request_stop()  # Stop the threads
            coord.join(threads)  # Wait for threads to stop

    @timeit
    def train(self):
        print("......begin training......")
        train_data_config = ImageCaptionDataConfig(model_name=self.model_name)

        model_config = ImageCaptionConfig(model_name=self.model_name, mode='train')

        with tf.Graph().as_default(), tf.Session(config=model_config.sess_config) as sess:
            initializer = tf.random_uniform_initializer(-model_config.initializer_scale,
                                                        model_config.initializer_scale)
            with tf.variable_scope(model_config.model_name, reuse=None, initializer=initializer):
                train_data_reader = ImageCaptionDataReader(data_config=train_data_config)
                model = ImageCaptionModel(config=model_config,
                                          data_reader=train_data_reader)
                # CheckPoint State
                checkpoint_state = tf.train.get_checkpoint_state(model.config.checkpoint_dir)
                if checkpoint_state:
                    self.logger.info("Loading model parameters from {}".format(checkpoint_state.model_checkpoint_path))
                    model.restore(sess, tf.train.latest_checkpoint(model.config.checkpoint_dir))
                else:
                    self.logger.info("Created model with fresh parameters.")
                    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                    sess.run(init_op)

                self.run_epoch(sess, model)

        print("......end training.....")

    def test(self):  #
        pass

    def inference(self):
        pass

    @timeit
    def get_logger(self):
        logger = logging.getLogger('tensorflow')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)
        self.logger = logger

    pass


def main(_):
    runner = ImageCaptionRunner()
    runner.train()


if __name__ == '__main__':
    tf.app.run()
