# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
from tensorflow.contrib import seq2seq, rnn
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import GRUCell, DropoutWrapper

from visual_caption.base.model.base_model import BaseModel
from visual_caption.utils.decorator_utils import timeit, define_scope


class ImageCaptionBiModel(BaseModel):
    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionBiModel, self).__init__(
            model_config, data_reader, mode)

    @timeit
    @define_scope(scope_name="inputs")
    def _build_inputs(self):
        data_type = self.model_config.data_type
        dim_visual_feature = self.model_config.data_config.dim_visual_feature

        if self.mode == ModeKeys.INFER:
            # In inference mode, images and inputs are fed via placeholders.
            self.image_feature = tf.placeholder(
                dtype=data_type, name='image_feature',
                shape=[None, dim_visual_feature])
            # self.region_features = tf.placeholder(
            #     dtype=data_type, name='region_features',
            #     shape=[None, None, dim_visual_feature])

            self.input_fw_feed = tf.placeholder(
                dtype=tf.int64, shape=[None], name="input_fw_feed")
            self.input_fw_seqs = tf.expand_dims(self.input_fw_feed, 1)

            self.input_bw_feed = tf.placeholder(
                dtype=tf.int64, shape=[None], name="input_bw_feed")
            self.input_bw_seqs = tf.expand_dims(self.input_bw_feed, 1)

        else:
            id_batch, width_batch, height_batch, depth_batch, feature_batch, \
            bbox_shape_batch, bbox_num, bbox_labels, bboxes, bbox_features, \
            caption_batch, fw_target_batch, bw_target_batch, \
            caption_ids, fw_target_ids, bw_target_ids, \
            caption_lengths, fw_target_lengths, bw_target_lengths = self.next_batch

            self.image_ids = id_batch

            self.image_feature = feature_batch
            self.region_features = bbox_features

            self.input_seqs = caption_ids
            self.fw_target_seqs = fw_target_ids
            self.bw_target_seqs = bw_target_ids

            self.input_lengths = caption_lengths
            self.fw_target_lengths = fw_target_lengths
            self.bw_target_lengths = bw_target_lengths

        # input visual features
        # expend_images = tf.expand_dims(self.image_feature, axis=1)
        # self.input_visual_features = tf.concat(
        #     values=[expend_images, self.region_features],
        #     name="input_visual_features", axis=1)

        # only use image_feature as visual_feature
        self.input_visual_features = self.image_feature

        # replace default model_config batch_size with data pipeline batch_size
        self.batch_size = tf.shape(self.image_feature)[0]
        # self.num_regions = tf.shape(self.input_visual_features)[0]
        # Maximum decoder time_steps in current batch
        if self.mode == ModeKeys.INFER:
            self.num_caption_max_length = self.model_config.data_config.num_caption_max_length
        else:
            self.num_caption_max_length = tf.reduce_max(self.input_lengths, axis=-1)

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
            if self.mode == ModeKeys.INFER:
                seq_fw_embeddings = tf.nn.embedding_lookup(
                    self.seq_embedding_map, self.input_fw_seqs)
                self.input_fw_seq_embeddings = seq_fw_embeddings

                seq_bw_embeddings = tf.nn.embedding_lookup(
                    self.seq_embedding_map, self.input_bw_seqs)
                self.input_bw_seq_embeddings = seq_bw_embeddings
            else:
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
    @define_scope(scope_name="graph")
    def _build_graph(self):
        self.__build_encoder()

        # # build attention model
        # self.__build_attention()

        # # compute attend visual features
        # self.__attend()

        self.__build_decoder()
        pass

    @timeit
    @define_scope(scope_name="encoder")
    def __build_encoder(self):
        # encode feature of given image and regions into
        pass

    @timeit
    def __create_rnn_cell(self):

        num_units = self.model_config.num_hidden_unit
        rnn_cell = rnn.GRUCell(num_units=num_units)
        if self.mode == ModeKeys.TRAIN:
            dropout_keep = self.model_config.dropout_keep_prob
            rnn_cell = rnn.DropoutWrapper(
                cell=rnn_cell,
                input_keep_prob=dropout_keep,
                output_keep_prob=dropout_keep)
        return rnn_cell

    @timeit
    @define_scope(scope_name="decoder")
    def __build_decoder(self):
        """
        decoder for caption outputs
        :return:
        """
        batch_size = self.batch_size
        data_type = self.model_config.data_type
        language_fw_rnn_cell = self.__create_rnn_cell()
        language_bw_rnn_cell = self.__create_rnn_cell()

        with tf.variable_scope("language", reuse=tf.AUTO_REUSE) as language_scope:

            language_fw_zero_state = language_fw_rnn_cell.zero_state(batch_size, data_type)
            language_bw_zero_state = language_bw_rnn_cell.zero_state(batch_size, data_type)

            # the combination of visual_feature and seq embeddings as the initial inputs
            language_fw_initial_input = tf.nn.l2_normalize(
                tf.concat([self.input_visual_embeddings, self.start_seq_embeddings], axis=-1),
                dim=-1, name="language_fw_initial_inputs")
            language_bw_initial_input = tf.nn.l2_normalize(
                tf.concat([self.input_visual_embeddings, self.end_seq_embeddings], axis=-1),
                dim=-1, name="language_bw_initial_inputs")

            # for initial state
            self.language_fw_initial_output, self.language_fw_initial_state = language_fw_rnn_cell(
                language_fw_initial_input, language_fw_zero_state)
            self.language_bw_initial_output, self.language_bw_initial_state = language_bw_rnn_cell(
                language_bw_initial_input, language_bw_zero_state)

        if self.mode == ModeKeys.INFER:
            # # use zero state as initial state
            self.language_fw_initial_state = language_fw_zero_state
            self.language_bw_initial_state = language_bw_zero_state

            # state feed for single step inference
            self.language_fw_state_feed = tf.placeholder(
                shape=[None, language_fw_rnn_cell.state_size],
                name="language_fw_state_feed", dtype=data_type)
            # inference backward state feed
            self.language_bw_state_feed = tf.placeholder(
                shape=[None, language_bw_rnn_cell.state_size],
                name="language_bw_state_feed", dtype=data_type)

            input_fw_seqs = self.input_fw_seq_embeddings
            input_bw_seqs = self.input_bw_seq_embeddings
            input_fw_seq_visuals = tf.tile(
                tf.expand_dims(self.input_visual_embeddings, axis=0),
                multiples=[tf.shape(input_fw_seqs)[0], 1, 1])
            input_bw_seq_visuals = tf.tile(
                tf.expand_dims(self.input_visual_embeddings, axis=0),
                multiples=[tf.shape(input_bw_seqs)[0], 1, 1])

            # for inference inputs
            language_fw_input = tf.squeeze(tf.nn.l2_normalize(
                tf.concat([input_fw_seq_visuals, input_fw_seqs], axis=-1),
                dim=-1), axis=1, name="language_fw_input")
            language_bw_input = tf.squeeze(tf.nn.l2_normalize(
                tf.concat([input_bw_seq_visuals, input_bw_seqs], axis=-1),
                dim=-1), axis=1, name="language_bw_input")

            # for single step inference
            language_fw_outputs, self.language_fw_new_state = language_fw_rnn_cell(
                inputs=language_fw_input, state=self.language_fw_state_feed)
            language_bw_outputs, self.language_bw_new_state = language_bw_rnn_cell(
                inputs=language_bw_input, state=self.language_bw_state_feed)

            # language_fw_outputs = tf.expand_dims(input=language_fw_outputs, dim=0)
            # language_bw_outputs = tf.expand_dims(input=language_bw_outputs, dim=0)
        else:
            input_seqs = self.input_seq_embeddings
            input_seq_visuals = tf.tile(
                tf.expand_dims(self.input_visual_embeddings, axis=1),
                multiples=[1, tf.shape(input_seqs)[1], 1])
            # inputs for language_model
            language_inputs = tf.nn.l2_normalize(
                tf.concat([input_seq_visuals, input_seqs], axis=-1),
                dim=-1, name="language_inputs")
            # unfold for language_model
            (language_fw_outputs, language_bw_outputs), language_states = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=language_fw_rnn_cell,
                    cell_bw=language_bw_rnn_cell,
                    inputs=language_inputs,
                    sequence_length=self.input_lengths,
                    initial_state_fw=language_fw_zero_state,
                    initial_state_bw=language_bw_zero_state,
                    dtype=data_type)

        self.language_fw_outputs = language_fw_outputs
        # self.language_bw_outputs = tf.reverse(language_bw_outputs,axis=[1])
        self.language_bw_outputs = language_bw_outputs

    @timeit
    def _build_loss(self):
        # Compute logits and weights
        vocab_size = self.data_reader.vocabulary.num_vocab

        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE) as logits_scope:
            fw_logits = tf.layers.dense(
                inputs=self.language_fw_outputs, units=vocab_size,
                kernel_initializer=self.initializer, name="fw_logits")
            bw_logits = tf.layers.dense(
                inputs=self.language_bw_outputs, units=vocab_size,
                kernel_initializer=self.initializer, name="bw_logits")
        self.fw_predictions = tf.cast(tf.argmax(fw_logits, axis=-1), tf.int32)
        self.bw_predictions = tf.cast(tf.argmax(bw_logits, axis=-1), tf.int32)

        if self.mode == ModeKeys.INFER:
            self.fw_softmax = tf.nn.softmax(fw_logits, name="fw_softmax")
            self.bw_softmax = tf.nn.softmax(bw_logits, name="bw_softmax")

            self.fw_predict = tf.cast(tf.argmax(fw_logits, axis=-1), tf.int32)
            self.bw_predict = tf.cast(tf.argmax(bw_logits, axis=-1), tf.int32)

        else:
            fw_weights = tf.sequence_mask(lengths=self.fw_target_lengths,
                                          dtype=self.model_config.data_type,
                                          name='masks')
            self.mask_weights = fw_weights
            bw_weights = tf.sequence_mask(lengths=self.bw_target_lengths,
                                          dtype=self.model_config.data_type,
                                          name='masks')
            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE) as loss_scope:
                fw_batch_loss = seq2seq.sequence_loss(
                    logits=fw_logits, targets=self.fw_target_seqs,
                    weights=fw_weights, name="fw_sequence_loss")
                bw_batch_loss = seq2seq.sequence_loss(
                    logits=bw_logits, targets=self.bw_target_seqs,
                    weights=bw_weights, name="bw_sequence_loss")
                self.fw_batch_loss = fw_batch_loss
                self.bw_batch_loss = bw_batch_loss

                tf.summary.scalar("fw_loss", self.fw_batch_loss)
                tf.summary.scalar("bw_loss", self.bw_batch_loss)

                # fw_shape = tf.shape(self.language_fw_outputs)
                # fw_size = [fw_shape[0], fw_shape[1]-2, fw_shape[2]]
                # fw_seqs = tf.slice(self.language_fw_outputs,
                #                    begin=[0, 0, 0], size=fw_size)
                # bw_shape = tf.shape(self.language_bw_outputs)
                # bw_size = [bw_shape[0], bw_shape[1]-2, bw_shape[2]]
                # bw_seqs = tf.slice(self.language_bw_outputs,
                #                    begin=[0, 2, 0], size=bw_size)
                # distance_loss = tf.losses.cosine_distance(fw_seqs, bw_seqs, dim=1)
                # tf.summary.scalar("distance_loss", distance_loss)
                # self.batch_loss = fw_batch_loss + bw_batch_loss + distance_loss

                self.batch_loss = fw_batch_loss + bw_batch_loss
                tf.summary.scalar("loss", self.batch_loss)

            with tf.variable_scope("accuracy", reuse=tf.AUTO_REUSE) as accuracy_scope:

                fw_correct_prediction = tf.equal(self.fw_predictions, self.fw_target_seqs)
                bw_correct_prediction = tf.equal(self.bw_predictions, self.bw_target_seqs)

                fw_batch_accuracy = tf.div(tf.reduce_sum(
                    tf.multiply(tf.cast(fw_correct_prediction, tf.float32), fw_weights)),
                    tf.reduce_sum(fw_weights))
                bw_batch_accuracy = tf.div(tf.reduce_sum(
                    tf.multiply(tf.cast(bw_correct_prediction, tf.float32), bw_weights)),
                    tf.reduce_sum(bw_weights))

                self.fw_batch_accuracy = fw_batch_accuracy
                tf.summary.scalar("fw_accuracy", self.fw_batch_accuracy)

                self.bw_batch_accuracy = bw_batch_accuracy
                tf.summary.scalar("bw_accuracy", self.bw_batch_accuracy)

    @timeit
    @define_scope(scope_name='optimizer')
    def _build_optimizer(self):
        config = self.model_config
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = 1.e-3
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            tf.summary.scalar('learning_rate', self.learning_rate)

    @timeit
    @define_scope(scope_name='gradients')
    def _build_gradients(self):
        """Clipping gradients of a model."""
        # if self.mode is not ModeKeys.INFER:
        #     trainables = tf.trainable_variables()
        #     with tf.device(self._get_gpu(self.model_config.nlanguage_bw_initial_inputum_gpus - 1)):
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
            #     trainables = tf.trainable_variables()
            #     grads_and_vars = zip(self._gradients, trainables)
            #     self.train_op = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars,
            #                                                    global_step=self.global_step_tensor,
            #                                                    name='train_step')
            self.train_op = self.optimizer.minimize(
                self.batch_loss, global_step=self.global_step_tensor)
