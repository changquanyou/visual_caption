# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import time

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from slim.nets.inception_resnet_v2 import slim
from visual_caption.base.base_runner import BaseRunner
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_reader import ImageCaptionDataReader
from visual_caption.image_caption.model.faster_rcnn_model import FasterRCNNModel
from visual_caption.image_caption.model.image_caption_config import ImageCaptionConfig
from visual_caption.utils import image_utils
from visual_caption.utils.decorator_utils import timeit


def get_init_fn(checkpoint_path):
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


class FasterRCNNModelRunner(BaseRunner):
    def __init__(self):
        super(FasterRCNNModelRunner, self).__init__()

        self.data_config = ImageCaptionDataConfig()
        self.data_reader = ImageCaptionDataReader(data_config=self.data_config)
        self.model_config = ImageCaptionConfig(data_config=self.data_config,
                                               model_name="image_caption_faster_rcnn")

        self.index2token = self.data_reader.data_embedding.index2token
        self.token2index = self.data_reader.data_embedding.token2index

        self.token_begin = self.model_config.data_config.token_begin
        self.token_end = self.model_config.data_config.token_end

        self.token_begin_id = self.token2index[self.token_begin]
        self.token_end_id = self.token2index[self.token_end]

        pass

    @timeit
    def train(self):
        model = FasterRCNNModel(model_config=self.model_config,
                                data_reader=self.data_reader,
                                mode=ModeKeys.TRAIN)
        fetches = [model.summary_merged, model.loss, model.accuracy, model.train_op,
                   model.image_ids, model.input_seqs, model.target_seqs, model.predictions]

        image_dir = self.model_config.data_config.train_image_dir

        format_string = "{0}: epoch={1:2d}, batch={2:6d}, batch_size={3:2d}, " \
                        "step={4:6d}, loss={5:.6f}, acc={6:.6f}, elapsed={7:.6f}"
        max_acc = 0.0
        with tf.Session(config=self.model_config.sess_config) as sess:

            if not model.restore_model(sess=sess):
                model.logger.info("Created model with fresh parameters.")
                init_op = tf.group(tf.local_variables_initializer(),
                                   tf.global_variables_initializer())
                sess.run(init_op)
                sess.run(tf.tables_initializer())
                print("begin to restore inception_resnet_v2 model ")
                checkpoint_path = self.model_config.inception_resnet_v2_ckpt
                get_init_fn(checkpoint_path=checkpoint_path)
                print("restored inception_resnet_v2 model successfully! ")

            else:
                # running first internal evaluation
                max_acc = self._internal_eval(model=model, sess=sess)
            model.summary_writer.add_graph(sess.graph)
            train_init_op = self.data_reader.get_train_init_op()
            begin = time.time()
            for epoch in range(model.model_config.max_max_epoch):
                sess.run(train_init_op)  # initial train data options
                step_begin = time.time()
                batch = 0
                while True:  # train each batch in a epoch
                    try:
                        batch_data = sess.run(model.next_batch)
                        (image_ids, image_features, captions, targets,
                         caption_ids, target_ids, caption_lengths, target_lengths) = batch_data

                        image_files = [os.path.join(image_dir, image_id.decode()) for image_id in image_ids]
                        input_images =  image_utils.load_images(image_files)
                        feed_dict = {model.image_ids: image_ids,
                                     model.input_images: input_images,
                                     model.input_seqs: caption_ids,
                                     model.input_lengths:caption_lengths,
                                     model.target_seqs: target_ids,
                                     model.target_lengths:target_lengths}

                        # run training step
                        result_batch = sess.run(fetches=fetches, feed_dict=feed_dict)

                        batch += 1
                        global_step = tf.train.global_step(sess, model.global_step_tensor)
                        # display and summarize training result
                        if batch % model.model_config.display_and_summary_step == 0:
                            batch_summary, loss, acc, _, image_ids, \
                            input_seqs, target_seqs, predicts = result_batch
                            batch_size = len(predicts)
                            # self._display_content(image_ids, input_seqs, predicts, target_seqs)
                            # add train summaries
                            model.summary_writer.add_summary(
                                summary=batch_summary, global_step=global_step)
                            print(format_string.format(model.mode, epoch, batch, batch_size,
                                                       global_step, loss, acc, time.time() - step_begin))
                            step_begin = time.time()
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


def main(_):
    runner = FasterRCNNModelRunner()
    runner.train()


if __name__ == '__main__':
    tf.app.run()
