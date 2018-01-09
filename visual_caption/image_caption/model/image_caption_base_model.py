# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from visual_caption.base.model.base_model import BaseModel
from visual_caption.utils.decorator_utils import timeit, define_scope


class ImageCaptionBaseModel(BaseModel):
    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionBaseModel, self).__init__(
            model_config, data_reader, mode)
        self.loss = None


    @timeit
    @define_scope(scope_name='inputs')
    def _build_inputs(self):
        data_type = self.model_config.data_type
        dim_visual_feature = self.model_config.data_config.dim_visual_feature
        if self.mode == ModeKeys.INFER:
            self.image_feature = tf.placeholder(dtype=data_type,
                                                shape=[None, dim_visual_feature],
                                                name="input_image")
            self.input_feed = tf.placeholder(dtype=tf.int64,
                                             shape=[None],  # batch_size
                                             name="input_feed")
            input_seqs = tf.expand_dims(self.input_feed, 1)
            self.input_seqs = input_seqs
        else:
            (image_id_batch, width_batch, height_batch, depth_batch, image_feature_batch,  # for image
             bbox_shape_batch, bbox_num_batch, bbox_labels, bboxes, bbox_features,  # for bbox
             caption_batch, fw_target_batch, bw_target_batch,  # for text
             caption_ids, fw_target_ids, bw_target_ids,  # for ids
             input_lengths) = self.next_batch

            self.image_ids = image_id_batch
            self.image_feature = image_feature_batch
            self.input_seqs = caption_ids
            self.target_seqs = fw_target_ids
            self.input_lengths = input_lengths

        # input visual features
        # expend_images = tf.expand_dims(self.image_feature, axis=1)
        # self.input_visual_features = tf.concat(name="input_visual_features", axis=1,
        #                                        values=[expend_images, self.region_features])

        # only use image_feature as visual_feature
        self.input_visual_features = self.image_feature

        # replace default model config batch_size with data pipeline batch_size
        self.batch_size = tf.shape(self.image_feature)[0]
        # Maximum decoder time_steps in current batch
        self.max_seq_length = self.model_config.length_max_output

        pass

    @timeit
    @define_scope(scope_name="embeddings")
    def _build_embeddings(self):

        self.data_config = self.model_config.data_config
        self.token_start_id = self.data_reader.vocabulary.start_id
        self.token_end_id = self.data_reader.vocabulary.end_id
        vocab_num = self.data_reader.vocabulary.num_vocab

        embedding_size = self.model_config.data_config.dim_token_feature
        # Save the embedding size in the graph.
        tf.constant(embedding_size, name="embedding_size")

        self.vocab_table = self.data_reader.vocab_table
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.seq_embedding_map = tf.get_variable(
                shape=[vocab_num + 1, embedding_size],
                dtype=self.model_config.data_type,
                initializer=self.emb_initializer,
                trainable=True,
                name='seq_embedding_map')
            seq_embeddings = tf.nn.embedding_lookup(
                self.seq_embedding_map, self.input_seqs)
        self.input_seq_embeddings = seq_embeddings

        # for token begin batch embeddings
        start_embedding = tf.nn.embedding_lookup(
            self.seq_embedding_map, [self.token_start_id])
        self.start_seq_embeddings = tf.tile(input=start_embedding,
                                            multiples=[self.batch_size, 1],
                                            name="start_seq_embeddings")
        # for token end batch embeddings
        end_embedding = tf.nn.embedding_lookup(
            self.seq_embedding_map, [self.token_end_id])
        self.end_seq_embeddings = tf.tile(input=end_embedding,
                                          multiples=[self.batch_size, 1],
                                          name="end_seq_embeddings")

        # Mapping visual features into embedding space.
        with tf.variable_scope("visual_embeddings") as scope:
            visual_embeddings = tf.layers.dense(
                inputs=self.input_visual_features,
                units=embedding_size)
        self.input_visual_embeddings = tf.nn.l2_normalize(
            visual_embeddings, dim=-1, name="input_visual_embeddings")
        pass


    @timeit
    @define_scope(scope_name='optimizer')
    def _build_optimizer(self):
        config = self.model_config
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            tf.summary.scalar('learning_rate', config.learning_rate)

    @timeit
    @define_scope(scope_name='gradients')
    def _build_gradients(self):
        """Clipping gradients of a model."""
        # if self.mode is not ModeKeys.INFER:
        #     trainables = tf.trainable_variables()
        #     with tf.device(self._get_gpu(self.model_config.num_gpus - 1)):
        #         gradients = tf.gradients(self.loss, trainables)
        #         # clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        #         #     gradients, self.model_config.max_grad_norm)
        #         self._gradients = gradients
        #         # tf.summary.scalar("grad_norm", gradient_norm)
        #         tf.summary.scalar("clipped_gradient", tf.global_norm(gradients))

    @timeit
    @define_scope(scope_name='train_op')
    def _build_train_op(self):
        if self.mode == ModeKeys.TRAIN:
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
