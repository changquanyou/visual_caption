# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import time

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from visual_caption.base.base_runner import BaseRunner
from visual_caption.image_caption.data.data_config import ImageCaptionFullDataConfig, ImageCaptionAttentionDataConfig
from visual_caption.image_caption.data.data_reader import ImageCaptionDataReader
from visual_caption.image_caption.model.image_caption_attention_model import ImageCaptionAttentionModel
from visual_caption.image_caption.model.image_caption_config import ImageCaptionConfig
from visual_caption.utils.decorator_utils import timeit

class ImageCaptionAttentionRunner(BaseRunner):
    def __init__(self):
        super(ImageCaptionAttentionRunner, self).__init__()

        self.data_config = ImageCaptionAttentionDataConfig()
        self.data_reader = ImageCaptionDataReader(data_config=self.data_config)
        self.model_config = ImageCaptionConfig(data_config=self.data_config,
                                               model_name=self.data_config.model_name)

        self.index2token = self.data_reader.data_embedding.index2token
        self.token2index = self.data_reader.data_embedding.token2index

        self.token_begin = self.model_config.data_config.token_begin
        self.token_end = self.model_config.data_config.token_end

        self.token_begin_id = self.token2index[self.token_begin]
        self.token_end_id = self.token2index[self.token_end]

    @timeit
    def train(self):
        model = ImageCaptionAttentionModel(model_config=self.model_config,
                                      data_reader=self.data_reader,
                                      mode=ModeKeys.TRAIN)
        fetches = [model.summary_merged, model.loss, model.accuracy, model.train_op,
                   model.image_ids, model.input_seqs, model.target_seqs, model.predictions]

        format_string = "{0}: epoch={1:2d}, batch={2:6d}, batch_size={3:2d}, " \
                        "step={4:6d}, loss={5:.6f}, acc={6:.6f}, elapsed={7:.6f}"
        with tf.Session(config=self.model_config.sess_config) as sess:
            model.summary_writer.add_graph(sess.graph)
            if not model.restore_model(sess=sess):
                model.logger.info("Created model with fresh parameters.")
                init_op = tf.group(tf.local_variables_initializer(),
                                   tf.global_variables_initializer())
                sess.run(init_op)
            sess.run(tf.tables_initializer())
            train_init_op = self.data_reader.get_train_init_op()
            begin = time.time()
            # running the first internal evaluation
            global_step = tf.train.global_step(sess, model.global_step_tensor)
            max_acc = 0.0
            if global_step > 0:
                max_acc = self._internal_eval(model=model, sess=sess)
            for epoch in range(model.model_config.max_max_epoch):
                sess.run(train_init_op)  # initial train data options
                step_begin = time.time()
                batch = 0
                while True:  # train each batch in a epoch
                    try:
                        result_batch = sess.run(fetches)  # run training step
                        global_step = tf.train.global_step(sess, model.global_step_tensor)
                        batch_summary, loss, acc, _, image_ids, input_seqs, \
                        target_seqs, predicts = result_batch
                        batch_size = len(predicts)

                        batch += 1
                        # display and summarize training result
                        if batch % model.model_config.display_and_summary_step == 0:
                            # self._display_content(image_ids, input_seqs, predicts, target_seqs)
                            # add train summaries
                            model.summary_writer.add_summary(
                                summary=batch_summary, global_step=global_step)
                            print(format_string.format(model.mode, epoch, batch, batch_size,
                                                       global_step, loss, acc, time.time() - step_begin))
                            step_begin = time.time()

                        if batch % 2000 == 0:
                            try:
                                valid_result = self._internal_eval(model=model, sess=sess)
                            except tf.errors.OutOfRangeError:
                                global_step = tf.train.global_step(sess,
                                                                   model.global_step_tensor)
                                model.logger.info("finished validation in training step {}"
                                                  .format(global_step))
                            valid_acc = valid_result
                            if valid_acc > max_acc:  # save the best model session
                                max_acc = valid_acc
                                model.save_model(sess=sess, global_step=global_step)
                                print('training: epoch={}, step={}, validation: average_result ={}'
                                      .format(epoch, global_step, valid_result))
                            print("training epoch={} finished with {} batches, global_step={}, elapsed={:.4f} "
                                  .format(epoch, batch, global_step, time.time() - begin))

                    except tf.errors.OutOfRangeError:  # ==> "End of training dataset"
                        try:
                            valid_result = self._internal_eval(model=model, sess=sess)
                        except tf.errors.OutOfRangeError:
                            global_step = tf.train.global_step(sess,
                                                               model.global_step_tensor)
                            model.logger.info("finished validation in training step {}"
                                              .format(global_step))
                        valid_acc = valid_result
                        if valid_acc > max_acc:  # save the best model session
                            max_acc = valid_acc
                            model.save_model(sess=sess, global_step=global_step)
                            print('training: epoch={}, step={}, validation: average_result ={}'
                                  .format(epoch, global_step, valid_result))
                        print("training epoch={} finished with {} batches, global_step={}, elapsed={:.4f} "
                              .format(epoch, batch, global_step, time.time() - begin))
                        break  # break the training while True
        pass
        # validation with current (such as training) session on validation data set



def main(_):
    runner = ImageCaptionAttentionRunner()
    runner.train()
    # runner.eval()
    # runner.infer()


if __name__ == '__main__':
    tf.app.run()