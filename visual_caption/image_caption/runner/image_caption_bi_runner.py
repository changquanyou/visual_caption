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
from visual_caption.image_caption.inference.image_caption_bi_generator import ImageCaptionBackwardGenerator, \
    ImageCaptionForwardGenerator
from visual_caption.image_caption.model.image_caption_bi_model import ImageCaptionBiModel
from visual_caption.image_caption.model.image_caption_config import ImageCaptionModelConfig
from visual_caption.utils.decorator_utils import timeit


class ImageCaptionBiRunner(BaseRunner):
    """
       image caption bidirectional model runner
    """

    def __init__(self):
        super(ImageCaptionBiRunner, self).__init__()
        self.data_config = ImageCaptionDataConfig()

        # for the consistency of forward and backward
        self.data_config.batch_size = 200
        self.data_reader = ImageCaptionDataReader(
            data_config=self.data_config)
        self.model_config = ImageCaptionModelConfig(
            data_config=self.data_config,
            model_name="image_caption_bi")

        self.index2token = self.data_reader.vocabulary.reverse_vocab
        self.token2index = self.data_reader.vocabulary.vocab

        self.token_start = self.data_config.token_start
        self.token_end = self.data_config.token_end
        self.token_pad = self.data_config.token_pad

        self.token_start_id = self.token2index[self.token_start]
        self.token_end_id = self.token2index[self.token_end]
        self.token_pad_id = self.token2index[self.token_pad]

    @timeit
    def train(self):
        model = ImageCaptionBiModel(
            model_config=self.model_config,
            data_reader=self.data_reader,
            mode=ModeKeys.TRAIN)
        fetches = [
            model.summary_merged, model.bw_batch_loss, model.train_op,
            model.image_ids,
            model.input_seqs, model.fw_target_seqs, model.bw_target_seqs,
            model.fw_predictions, model.bw_predictions,
            model.fw_batch_accuracy, model.bw_batch_accuracy,
            model.mask_weights, model.input_lengths
        ]
        format_string = "{0}: epoch={1:2d}, batch={2:4d}, batch_size={3:2d}, step={4:6d}," \
                        " loss={5:.4f}, fw_acc={6:.4f}, bw_acc={7:.4f}, elapsed={8:.4f}"
        display_step = model.model_config.display_and_summary_step
        display_step = 20
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

                        batch_summary, loss, train_op, \
                        image_ids, input_seqs, fw_target_seqs, bw_target_seqs, \
                        fw_predicts, bw_predicts, fw_accuracy, bw_accuracy, \
                        weights, input_lengths = result_batch
                        if global_step % display_step == 0:
                            # display and summarize training result
                            batch += 1
                            model.summary_writer.add_summary(
                                summary=batch_summary, global_step=global_step)
                            batch_size = len(image_ids)
                            print(format_string.format(
                                model.mode, epoch, batch, batch_size, global_step,
                                loss, fw_accuracy, bw_accuracy, time.time() - step_begin))
                            step_begin = time.time()
                        if global_step % 2000 == 0:
                            self._display_results(
                                image_ids=image_ids, inputs=input_seqs,
                                fw_targets=fw_target_seqs, bw_targets=bw_target_seqs,
                                fw_predicts=fw_predicts, bw_predicts=bw_predicts,
                                fw_accuracy=fw_accuracy, bw_accuracy=bw_accuracy,
                                weights=weights, input_lengths=input_lengths
                            )
                            try:
                                valid_result = self._internal_eval(model=model, sess=sess)
                            except tf.errors.OutOfRangeError:
                                global_step = tf.train.global_step(
                                    sess, model.global_step_tensor)
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
                            global_step = tf.train.global_step(
                                sess, model.global_step_tensor)
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

    def _internal_eval(self, model, sess):
        """
        running internal evaluation with current sess
        :param model:
        :param sess:
        :return:
       　"""
        fetches = [model.bw_batch_accuracy, model.bw_batch_loss, model.summary_merged]
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
                eval_acc += acc
                batch_count += 1
                if batch_count % self.model_config.display_and_summary_step == 0:
                    model.summary_valid_writer.add_summary(
                        summary=summaries, global_step=global_step)
                    print("valid: step={0:8d}, batch={1:4d} loss={2:.4f}, acc={3:.4f}, elapsed={4:.4f}"
                          .format(global_step, batch_count, loss, acc, time.time() - step_begin))
                    step_begin = time.time()
                if batch_count >= 100:
                    break
            except tf.errors.OutOfRangeError:  # ==> "End of validation dataset"
                print("_internal_eval finished : step={0}, batch={1}, elapsed={2:.4f}"
                      .format(global_step, batch_count, time.time() - step_begin))
                break

        if batch_count > 0:
            eval_acc = eval_acc / batch_count
        result = eval_acc
        return result
        pass

        pass

    def valid(self):
        pass

    def _get_sequence(self, seq_ids, length):
        seq_text = [self.index2token[id] for idx, id in enumerate(seq_ids) if idx < length]
        # if idx != self.token_end_id and idx != self.token_pad_id and idx < length]
        return seq_text

    def _display_results(self, image_ids, inputs=None,
                         fw_targets=None, bw_targets=None,
                         fw_predicts=None, bw_predicts=None,
                         fw_accuracy=None, bw_accuracy=None,
                         weights=None, input_lengths=None):
        # for idx, image_id in enumerate(image_ids):
        if len(inputs) > 0:
            idx = -1
            image_id = image_ids[idx]
            length = input_lengths[idx]
            print("image_id={}, length={}, fw_accuracy={:.4f}, bw_accuracy={:.4f}".
                  format(image_id, length, fw_accuracy, bw_accuracy))
            print("\t     input={}".format(self._get_sequence(inputs[idx], length)))
            print("\t fw_target={}".format(self._get_sequence(fw_targets[idx], length)))
            print("\t bw_target={}".format(self._get_sequence(bw_targets[idx], length)))
            print("\tfw_predict={}".format(self._get_sequence(fw_predicts[idx], length)))
            print("\tbw_predict={}".format(self._get_sequence(bw_predicts[idx], length)))
            print("\t    weight={}".format(weights[idx]))

    def infer(self):
        infer_model = ImageCaptionBiModel(
            model_config=self.model_config,
            data_reader=self.data_reader,
            mode=ModeKeys.INFER)
        model = infer_model
        # Initialize beam search Caption Generator
        fw_generator = ImageCaptionForwardGenerator(
            model=model, vocab=self.token2index, token_start=self.token_start,
            token_end=self.token_end, beam_size=self.data_config.beam_size,
            max_caption_length=self.data_config.num_caption_max_length)

        bw_generator = ImageCaptionBackwardGenerator(
            model=model, vocab=self.token2index, token_start=self.token_start,
            token_end=self.token_end, beam_size=self.data_config.beam_size,
            max_caption_length=self.data_config.num_caption_max_length)

        with tf.Session(config=model.model_config.sess_config) as sess:
            if model.restore_model(sess):
                sess.run(tf.tables_initializer())
                begin = time.time()
                dataset_init_op = model.data_reader.get_valid_init_op()
                sess.run(dataset_init_op)  # initial train data options
                global_step = tf.train.global_step(sess, model.global_step_tensor)
                batch = 0
                while True:  # train each batch in a epoch
                    try:
                        batch_data = sess.run(model.next_batch)
                        (id_batch, width_batch, height_batch, depth_batch, feature_batch,
                         bbox_shape_batch, bbox_num, bbox_labels, bboxes, bbox_features,
                         caption_batch, fw_target_batch, bw_target_batch,
                         caption_ids, fw_target_ids, bw_target_ids,
                         caption_lengths, fw_target_lengths, bw_target_lengths) = batch_data

                        for idx, image_id in enumerate(id_batch):  # for each image
                            image_feature = feature_batch[idx].reshape(1, -1)
                            print("image_id={}".format(image_id))
                            fw_predict_captions = fw_generator.beam_search(
                                sess=sess, image_feature=image_feature)
                            bw_predict_captions = bw_generator.beam_search(
                                sess=sess, image_feature=image_feature)
                            caption_length = caption_lengths[idx]
                            caption_text = b' '.join(caption_batch[idx][:caption_length]).decode('utf-8')
                            print("target_caption: {}".format(caption_text))
                            print("forward:----------------------------------------")
                            for i in range(self.data_config.beam_size):
                                caption = fw_predict_captions[i]
                                caption_text = [self.index2token[id] for id in caption.sentence]
                                print("\tfw_beam_idx:{:1d}, logprob:{:4.4f}, caption: {}"
                                      .format(idx, caption.logprob, " ".join(caption_text)))
                            print("backward:----------------------------------------")
                            for i in range(self.data_config.beam_size):
                                caption = bw_predict_captions[i]
                                caption_text = [self.index2token[id] for id in reversed(caption.sentence)]
                                print("\tbw_beam_idx:{:1d}, logprob:{:4.4f}, caption: {}"
                                      .format(idx, caption.logprob, " ".join(caption_text)))

                    except tf.errors.OutOfRangeError:  # ==> "End of training dataset"
                        print(" finished with {} batches, global_step={}, elapsed={} "
                              .format(batch, global_step, time.time() - begin))
                        break  # break the training while True

        pass

    def infer_fw_greedy(self):
        infer_model = ImageCaptionBiModel(model_config=self.model_config,
                                          data_reader=self.data_reader,
                                          mode=ModeKeys.INFER)
        model = infer_model
        with tf.Session(config=model.model_config.sess_config) as sess:
            if model.restore_model(sess):
                sess.run(tf.tables_initializer())
                begin = time.time()
                dataset_init_op = model.data_reader.get_infer_init_op()
                sess.run(dataset_init_op)  # initial train data options
                global_step = tf.train.global_step(sess, model.global_step_tensor)
                batch = 0
                while True:  # train each batch in a epoch
                    try:
                        batch_data = sess.run(model.next_batch)
                        (id_batch, width_batch, height_batch, depth_batch, feature_batch,
                         bbox_shape_batch, bbox_num, bbox_labels, bboxes, bbox_features,
                         caption_batch, fw_target_batch, bw_target_batch,
                         caption_ids, fw_target_ids, bw_target_ids,
                         caption_lengths, fw_target_lengths, bw_target_lengths) = batch_data
                        for idx, image_id in enumerate(id_batch):  # for each image
                            image_feature = feature_batch[idx].reshape(1, -1)
                            predict_caption_ids = self._decode_fw_greedy(
                                model=model, sess=sess, image_feature=image_feature)
                            print("image_id={}".format(image_id))
                            caption_text = [self.index2token[idx] for idx in predict_caption_ids]
                            print("\tcaption:{}".format(" ".join(caption_text)))
                    except tf.errors.OutOfRangeError:  # ==> "End of training dataset"
                        print(" finished with {} batches, global_step={}, elapsed={} "
                              .format(batch, global_step, time.time() - begin))
                        break  # break the training while True

    def _decode_fw_greedy(self, model, sess, image_feature):
        fetches = [model.language_fw_initial_state]
        feed_dict = {model.image_feature: image_feature}
        language_fw_initial_state = sess.run(
            fetches=fetches, feed_dict=feed_dict)
        input_feed = [self.token_start_id]
        state_feed = language_fw_initial_state[0]
        fetches = [model.fw_predict, model.language_fw_new_state]
        predict_caption_ids = list()
        for idx in range(self.data_config.num_caption_max_length):
            feed_dict = {
                model.input_fw_feed: input_feed,
                model.language_fw_state_feed: state_feed,
                model.image_feature: image_feature
            }
            fw_predict, new_language_fw_state = sess.run(
                fetches=fetches, feed_dict=feed_dict)
            input_feed = fw_predict
            state_feed = new_language_fw_state
            if self.token_end_id == fw_predict[0]:
                break
            predict_caption_ids.append(fw_predict[0])
        return predict_caption_ids

    def infer_bw_greedy(self):
        infer_model = ImageCaptionBiModel(
            model_config=self.model_config,
            data_reader=self.data_reader,
            mode=ModeKeys.INFER)
        model = infer_model
        with tf.Session(config=model.model_config.sess_config) as sess:
            if model.restore_model(sess) is False:
                return
            sess.run(tf.tables_initializer())
            begin = time.time()
            dataset_init_op = model.data_reader.get_infer_init_op()
            sess.run(dataset_init_op)  # initial train data options
            global_step = tf.train.global_step(sess, model.global_step_tensor)
            batch = 0
            while True:  # train each batch in a epoch
                try:
                    batch_data = sess.run(model.next_batch)

                    (id_batch, width_batch, height_batch, depth_batch, feature_batch,
                     bbox_shape_batch, bbox_num, bbox_labels, bboxes, bbox_features,
                     caption_batch, fw_target_batch, bw_target_batch,
                     caption_ids, fw_target_ids, bw_target_ids,
                     caption_lengths, fw_target_lengths, bw_target_lengths) = batch_data

                    for idx, image_id in enumerate(id_batch):  # for each image
                        image_feature = feature_batch[idx].reshape(1, -1)
                        predict_caption_ids = self._decode_bw_greedy(
                            model=model, sess=sess, image_feature=image_feature)
                        print("image_id={}".format(image_id))
                        caption_text = [self.index2token[idx] for idx in reversed(predict_caption_ids)]
                        print("\tcaption:{}".format(" ".join(caption_text)))
                except tf.errors.OutOfRangeError:  # ==> "End of training dataset"
                    print(" finished with {} batches, global_step={}, elapsed={} "
                          .format(batch, global_step, time.time() - begin))
                    break  # break the training while True

    def _decode_bw_greedy(self, model, sess, image_feature):
        language_bw_initial_state = sess.run(
            fetches=[model.language_bw_initial_state],
            feed_dict={model.image_feature: image_feature})
        input_feed = [self.token_end_id]
        state_feed = language_bw_initial_state[0]
        fetches = [model.bw_predict, model.language_bw_new_state]
        predict_caption_ids = list()
        for idx in range(self.data_config.num_caption_max_length):
            feed_dict = {
                model.input_bw_feed: input_feed,
                model.language_bw_state_feed: state_feed,
                model.image_feature: image_feature
            }
            bw_predict, new_language_bw_state = sess.run(
                fetches=fetches, feed_dict=feed_dict)
            input_feed = bw_predict
            state_feed = new_language_bw_state
            if bw_predict[0] == self.token_start_id:
                break
            predict_caption_ids.append(bw_predict[0])
        return predict_caption_ids


def main(_):
    runner = ImageCaptionBiRunner()
    runner.train()
    # runner.eval()
    # runner.infer()
    # runner.infer_fw_greedy()
    # runner.infer_bw_greedy()


if __name__ == '__main__':
    tf.app.run()
