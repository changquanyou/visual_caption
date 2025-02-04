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
from visual_caption.image_caption.inference.caption_attention_generator import CaptionAttentionGenerator
from visual_caption.image_caption.model.image_caption_attention_model import ImageCaptionAttentionModel
from visual_caption.image_caption.model.image_caption_config import ImageCaptionModelConfig
from visual_caption.utils.decorator_utils import timeit


class ImageCaptionAttentionRunner(BaseRunner):
    """Image Caption Attention Model Runner
    """

    def __init__(self):
        super(ImageCaptionAttentionRunner, self).__init__()

        self.data_config = ImageCaptionDataConfig()

        # for the consistency of forward and backward
        self.data_config.batch_size = 200
        self.data_reader = ImageCaptionDataReader(
            data_config=self.data_config)
        self.model_config = ImageCaptionModelConfig(
            data_config=self.data_config,
            model_name="image_caption_attention")

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
        model = ImageCaptionAttentionModel(
            model_config=self.model_config,
            data_reader=self.data_reader,
            mode=ModeKeys.TRAIN)
        fetches = [
            model.summary_merged, model.loss,
            model.accuracy, model.train_op,
            model.image_ids, model.input_seqs,
            model.fw_target_seqs, model.predicts,
            model.mask_weights, model.input_lengths
        ]
        format_string = "{0}: epoch={1:2d}, batch={2:4d}, batch_size={3:2d}, " \
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
                        (batch_summary, loss, acc, _, image_ids, input_seqs,
                         target_seqs, predicts, weights, input_lengths) = result_batch
                        batch_size = len(predicts)
                        batch += 1
                        # display and summarize training result
                        if batch % model.model_config.display_and_summary_step == 0:
                            # add train summaries
                            model.summary_writer.add_summary(
                                summary=batch_summary, global_step=global_step)
                            print(format_string.format(model.mode, epoch, batch, batch_size,
                                                       global_step, loss, acc, time.time() - step_begin))
                            step_begin = time.time()
                        if global_step % 200 == 0 and global_step > 0:
                            self._display_results(
                                image_ids=image_ids, inputs=input_seqs, targets=target_seqs,
                                predicts=predicts, weights=weights, lengths=input_lengths)
                            try:
                                valid_result = self._internal_eval(model=model, sess=sess)
                            except tf.errors.OutOfRangeError:
                                global_step = tf.train.global_step(sess, model.global_step_tensor)
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

    def eval(self):
        pass

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
                eval_acc += acc
                batch_count += 1
                if batch_count % self.model_config.display_and_summary_step == 0:
                    model.summary_valid_writer.add_summary(
                        summary=summaries, global_step=global_step)
                    print("valid: step={0:8d}, batch={1} loss={2:.4f}, acc={3:.4f}, elapsed={4:.4f}"
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

    def infer(self):
        infer_model = ImageCaptionAttentionModel(model_config=self.model_config,
                                                 data_reader=self.data_reader,
                                                 mode=ModeKeys.INFER)
        model = infer_model
        # Initialize beam search Caption Generator
        caption_max_length = self.data_config.num_caption_max_length
        generator = CaptionAttentionGenerator(
            model=model, vocab=self.token2index,
            token_start=self.token_start, token_end=self.token_end,
            beam_size=self.data_config.beam_size,
            max_caption_length=caption_max_length
        )

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
                        (image_id_batch, width_batch, height_batch, depth_batch, image_feature_batch,  # for image
                         bbox_shape_batch, bbox_num_batch, bbox_labels, bboxes, bbox_features,  # for bbox
                         caption_batch, fw_target_batch, bw_target_batch,  # for text
                         caption_ids, fw_target_ids, bw_target_ids,  # for ids
                         input_lengths, fw_target_lengths, bw_target_lengths) = batch_data
                        for idx, image_id in enumerate(image_id_batch):  # for each image
                            image_feature = image_feature_batch[idx].reshape(1, -1)
                            region_features = bbox_features[idx].reshape(1, 36, -1)
                            print("image_id={}".format(image_id))
                            # predict multiple captions
                            predict_captions = generator.beam_search(
                                sess, image_feature, region_features)

                            for index, predict_caption in enumerate(predict_captions):
                                # convert each caption_ids into caption_texts
                                caption_text = [self.index2token[idx] for idx in predict_caption.sentence]
                                print("beam_idx:{:1d}, logprob:{:.4f}, caption:{}"
                                      .format(index, predict_caption.logprob, " ".join(caption_text)))
                                step_begin = time.time()
                    except tf.errors.OutOfRangeError:  # ==> "End of training dataset"
                        print(" finished with {} batches, global_step={}, elapsed={} "
                              .format(batch, global_step, time.time() - begin))
                        break  # break the training while True

        pass

    def _get_sequence(self, seq_ids, length):
        seq_text = [self.index2token[id] for idx, id in enumerate(seq_ids) if idx < length]
        # if idx != self.token_end_id and idx != self.token_pad_id and idx < length]
        return seq_text

    def _display_results(self, image_ids,
                         inputs=None, targets=None,
                         predicts=None, weights=None,
                         lengths=None):
        # for idx, image_id in enumerate(image_ids):
        idx = -1
        image_id = image_ids[idx]
        print("image_id={}".format(image_id))
        if len(inputs) > 0:
            length = lengths[idx]
            input = inputs[idx]
            print("\t  input={}".format(self._get_sequence(input, length)))
            target = targets[idx]
            print("\t target={}".format(self._get_sequence(target, length)))
            predict = predicts[idx]
            print("\tpredict={}".format(self._get_sequence(predict, length)))
            weight = weights[idx]
            print("\t weight={}".format(weight[:length]))


def main(_):
    runner = ImageCaptionAttentionRunner()
    runner.train()
    # runner.eval()
    # runner.infer()


if __name__ == '__main__':
    tf.app.run()
