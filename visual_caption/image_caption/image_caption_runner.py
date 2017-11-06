# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import time

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from visual_caption.base.base_runner import BaseRunner
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_reader import ImageCaptionDataReader
from visual_caption.image_caption.model.image_caption_config import ImageCaptionConfig
from visual_caption.image_caption.model.image_caption_model import ImageCaptionModel
from visual_caption.utils.decorator_utils import timeit


class ImageCaptionRunner(BaseRunner):
    def __init__(self):
        super(ImageCaptionRunner, self).__init__()

        data_config = ImageCaptionDataConfig()
        self.data_reader = ImageCaptionDataReader(data_config=data_config)
        self.model_config = ImageCaptionConfig(data_config=data_config,
                                               model_name=data_config.model_name)
        pass

    @timeit
    def train(self):
        model = ImageCaptionModel(model_config=self.model_config,
                                  data_reader=self.data_reader,
                                  mode=ModeKeys.TRAIN)

        fetches = [model.summary_merged,
                   model.loss, model.accuracy,
                   model.train_op, model.input_seqs]
        format_string = "{0}: epoch={1:2d}, batch={2:6d}, step={3:6d}," \
                        " loss={4:.6f}, acc={5:.6f}, elapsed={6:.6f}"
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

            # running first internal evaluation
            max_acc = self._internal_eval(model=model, sess=sess)
            for epoch in range(model.model_config.max_max_epoch):
                sess.run(train_init_op)  # initial train data options
                step_begin = time.time()
                batch = 0
                while True:  # train each batch in a epoch
                    try:
                        result_batch = sess.run(fetches)  # run training step
                        global_step = tf.train.global_step(sess, model.global_step_tensor)
                        # display and summarize training result
                        if global_step % model.model_config.display_and_summary_step == 0:
                            batch_summary, loss, acc, _, phrases = result_batch
                            # add train summaries
                            model.summary_writer.add_summary(
                                summary=batch_summary, global_step=global_step)
                            print(format_string.format(model.mode, epoch, batch, global_step, loss, acc,
                                                       time.time() - step_begin))
                            step_begin = time.time()
                        batch += 1

                        # # run internal_eval during training epoch
                        # if global_step % self.model_config.valid_step == 0 and global_step > 0:
                        #     try:
                        #         valid_result = self._internal_eval(sess=sess)
                        #     except tf.errors.OutOfRangeError:
                        #         self.logger.info("finished validation in training step {}".format(global_step))
                        #
                        #     valid_acc = valid_result
                        #     if valid_acc > max_acc:  # save the best model session
                        #         self._save_model(sess=sess, global_step=global_step)
                        #     print('training: epoch={}, step={}, validation: average_result ={}'
                        #           .format(epoch, global_step, valid_result))
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
                        print("training epoch={} finished with {} batches, global_step={}, elapsed={} "
                              .format(epoch, batch, global_step, time.time() - begin))
                        break  # break the training while True
        pass

        # validation with current (such as training) session on validation data set

    @timeit
    def _internal_eval(self, model, sess):
        """
        running internal evaluation with current sess
        :param model:
        :param sess:
        :return:
        """
        fetches = [model.accuracy, model.loss, model.summary_merged]
        batch_count = 0
        eval_acc = 0.0
        validation_init_op = self.data_reader.get_valid_init_op()
        # initialize validation dataset
        sess.run(validation_init_op)
        step_begin = time.time()
        global_step = tf.train.global_step(sess, model.global_step_tensor)
        while True:  # iterate eval batch at step
            try:
                eval_step_result = sess.run(fetches=fetches)
                acc, loss, summaries = eval_step_result
                model.summary_validation_writer.add_summary(
                    summary=summaries, global_step=global_step)
                eval_acc += acc
                batch_count += 1
                if batch_count % self.model_config.display_and_summary_step == 0:
                    print("valid: step={0:8d}, batch={1} loss={2:.4f}, acc={3:.4f}, elapsed={4:.4f}"
                          .format(global_step, batch_count, loss, acc, time.time() - step_begin))
                    step_begin = time.time()
                if batch_count >= 50:
                    break
            except tf.errors.OutOfRangeError:  # ==> "End of validation dataset"
                print("validation finished : step={0}, batch={1}, elapsed={2:.4f}"
                      .format(global_step, batch_count, time.time() - step_begin))
                break

        if batch_count > 0:
            eval_acc = eval_acc / batch_count
        result = eval_acc
        return result
        pass

    @timeit
    def valid(self):
        pass

    @timeit
    def infer(self):
        infer_model = ImageCaptionModel(model_config=self.model_config,
                                        data_reader=self.data_reader,
                                        mode=ModeKeys.INFER)
        index2token = self.data_reader.data_embedding.index2token

        model = infer_model
        fetches = [model.loss, model.accuracy,
                   model.image_ids,
                   model.input_seqs, model.target_seqs,
                   model.predict_predictions]
        format_string = "{}: batch={:6d}, step={:6d}, loss={:.6f}, acc={:.6f}, elapsed={:.6f}"
        with tf.Session(config=model.model_config.sess_config) as sess:
            model.summary_writer.add_graph(sess.graph)
            # CheckPoint State
            if not model.restore_model(sess=sess):
                init_op = tf.group(tf.local_variables_initializer(),
                                   tf.global_variables_initializer())
                sess.run(init_op)

            sess.run(tf.tables_initializer())
            begin = time.time()
            infer_init_op = model.data_reader.get_valid_init_op()
            sess.run(infer_init_op)  # initial infer data options
            batch = 0
            global_step = tf.train.global_step(sess, model.global_step_tensor)
            while True:  # train each batch in a epoch
                try:
                    batch_data = sess.run(model.next_batch)
                    (image_ids, image_features, captions, targets, caption_ids, target_ids,
                     caption_lengths, target_lengths) = batch_data
                    feed_dict = {

                        model.image_ids: image_ids,
                        model.input_image_embeddings: image_features,

                        model.input_seqs: caption_ids,
                        model.target_seqs: target_ids,

                        model.input_lengths: caption_lengths,
                        model.target_lengths: target_lengths,

                    }
                    result_batch = sess.run(fetches=fetches, feed_dict=feed_dict)  # run training step
                    batch += 1
                    global_step = tf.train.global_step(sess, model.global_step_tensor)
                    # display and summarize training result
                    loss, acc, image_ids, input_seqs, target_seqs, predicts = result_batch
                    for idx, image_id in enumerate(image_ids):
                        caption_text = b' '.join(captions[idx])
                        caption_text = caption_text.decode()

                        predict = predicts[idx]
                        predict_byte_list = [index2token[token_id] for token_id in predict]
                        predict_text = ' '.join(predict_byte_list)

                        print("image_id={}, \n\tcaption=[{}]\n\tpredict=[{}]"
                              .format(image_ids[idx], caption_text, predict_text))
                    # add train summaries
                    # print(format_string.format(model.mode, batch, global_step, loss, acc,
                    #                            time.time() - step_begin))
                    step_begin = time.time()
                except tf.errors.OutOfRangeError:  # ==> "End of training dataset"
                    print(" finished with {} batches, global_step={}, elapsed={} "
                          .format(batch, global_step, time.time() - begin))
                    break  # break the training while True

        pass


def main(_):
    runner = ImageCaptionRunner()
    runner.train()
    # runner.infer()


if __name__ == '__main__':
    tf.app.run()
