# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib import rnn
from tensorflow.contrib.learn import ModeKeys

from visual_caption.image_caption.model.image_caption_base_model import ImageCaptionBaseModel
from visual_caption.utils.decorator_utils import timeit, define_scope


class ImageCaptionAttentionModel(ImageCaptionBaseModel):
    """
        Image Caption Model with the below mechanism:
            1.Bottom-Up and Top-Down Attention
            2.Multi-modal Factorized Bilinear Pooling with Co-Attention

    """

    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionAttentionModel, self).__init__(
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
            self.region_features = tf.placeholder(
                dtype=data_type, name='region_features',
                shape=[None, None, dim_visual_feature])
            self.input_feed = tf.placeholder(
                dtype=tf.int64, shape=[None], name="input_feed")
            self.input_seqs = tf.expand_dims(self.input_feed, 1)

        else:
            (id_batch, width_batch, height_batch, depth_batch, feature_batch,
             bbox_shape_batch, bbox_num, bbox_labels, bboxes, bbox_features,
             caption_batch, fw_target_batch, bw_target_batch,
             caption_ids, fw_target_ids, bw_target_ids,
             caption_lengths, fw_target_lengths, bw_target_lengths) = self.next_batch

            self.image_ids = id_batch

            self.image_feature = feature_batch
            self.region_features = bbox_features

            self.input_seqs = caption_ids
            self.input_lengths = caption_lengths

            self.fw_target_seqs = fw_target_ids
            self.fw_target_lengths = fw_target_lengths

            self.bw_target_seqs = bw_target_ids
            self.bw_target_lengths = bw_target_lengths

            # input visual features
        expend_images = tf.expand_dims(self.image_feature, axis=1)

        self.input_visual_features = tf.concat(
            values=[expend_images, self.region_features],
            name="input_visual_features", axis=1)
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
            self.input_seq_embeddings = tf.nn.embedding_lookup(
                self.seq_embedding_map, self.input_seqs)

            # for token_start and token_end batch embeddings
            start_embedding = tf.nn.embedding_lookup(
                self.seq_embedding_map, [self.token_start_id])
            self.start_seq_embeddings = tf.tile(
                input=start_embedding, multiples=[self.batch_size, 1],
                name="start_seq_embeddings")
            end_embedding = tf.nn.embedding_lookup(
                self.seq_embedding_map, [self.token_end_id])
            self.end_seq_embeddings = tf.tile(
                input=end_embedding, multiples=[self.batch_size, 1],
                name="end_seq_embeddings")

        # Mapping visual features into embedding space.
        with tf.variable_scope("visual_embeddings") as visual_embedding_scope:
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

        # build attention model
        self.__build_attention()

        # compute attend visual features
        self.__attend()

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
    def __build_attention(self):
        """
        build attention model
        :return:
        """
        batch_size = self.batch_size
        data_type = self.model_config.data_type

        with tf.variable_scope("attention",
                               initializer=self.initializer,
                               reuse=tf.AUTO_REUSE) as attend_scope:
            attend_rnn_cell = self.__create_rnn_cell()

            # attend rnn_cell initial, shape=[batch,dim_visual_embedding]
            average_visuals = tf.reduce_mean(self.input_visual_embeddings,
                                             axis=1, name="average_visuals")
            # initial token attend input
            attend_initial_inputs = tf.nn.l2_normalize(
                tf.concat([average_visuals, self.start_seq_embeddings], axis=-1),
                name="attend_fw_initial_inputs", dim=-1)
            attend_zero_state = attend_rnn_cell.zero_state(
                batch_size=batch_size, dtype=data_type)
            attend_initial_outputs, attend_initial_state = attend_rnn_cell(
                attend_initial_inputs, attend_zero_state)

            # expend and tile visual_feature according with input seq
            seq_length = tf.shape(self.input_seq_embeddings)[1]
            seq_average_visuals = tf.tile(tf.expand_dims(average_visuals, axis=1),
                                          multiples=[1, seq_length, 1])

            attend_inputs = tf.concat([seq_average_visuals, self.input_seq_embeddings], axis=-1)
            attend_inputs = tf.nn.l2_normalize(attend_inputs, name="attend_inputs", dim=-1)

            if self.mode == ModeKeys.INFER:
                self.attend_initial_state = attend_zero_state
                self.attend_state_feed = tf.placeholder(
                    shape=[None, attend_rnn_cell.state_size],
                    name="attend_state_feed", dtype=data_type)
                attend_inputs = tf.squeeze(attend_inputs, axis=1)
                attend_outputs, self.attend_new_state = attend_rnn_cell(
                    inputs=attend_inputs, state=self.attend_state_feed)
                attend_outputs = tf.expand_dims(input=attend_outputs, axis=0)

            else:
                attend_outputs, attend_states = tf.nn.dynamic_rnn(
                    cell=attend_rnn_cell, inputs=attend_inputs,
                    sequence_length=self.input_lengths,
                    initial_state=attend_zero_state,
                    dtype=data_type)

        self.attend_initial_outputs = attend_initial_outputs
        self.attend_outputs = attend_outputs

        pass

    @timeit
    @define_scope(scope_name="attend")
    def __attend(self):
        """
                compute the attend weighted visual features based on attend_outputs
               """
        data_type = self.model_config.data_type
        num_visual_features = tf.shape(self.input_visual_embeddings)[1]
        dim_hidden = self.model_config.dim_fused_feature

        # expand to sequence length,
        # shape = [batch, seq_length, num_regions, dim_visual_embedding]
        # expand to sequence length,
        # shape = [batch, seq_length, num_regions, dim_visual_embedding]
        seq_visual_embeddings = tf.tile(
            tf.expand_dims(self.input_visual_embeddings, axis=1),
            multiples=[1, tf.shape(self.input_seqs)[1], 1, 1],
            name="seq_visual_embeddings")
        # mapping seq_visual_embeddings into hidden space,
        # shape = [batch, seq_length, num_regions, dim_hidden]
        dense_v_a = tf.layers.dense(
            inputs=seq_visual_embeddings, units=dim_hidden,
            name="seq_visual_embeddings_mapping")

        # attend_outputs, expand to num_regions,
        # shape=[batch,　seq_length,　num_regions,　dim_attend_feature]
        seq_attend_embeddings = tf.tile(
            tf.expand_dims(self.attend_outputs, axis=2),
            multiples=[1, 1, num_visual_features, 1],
            name="seq_attend_embeddings")

        # mapping seq_attend_embeddings into hidden space,
        # shape = [batch, seq_length, num_regions, dim_hidden]
        dense_h_a = tf.layers.dense(
            inputs=seq_attend_embeddings, units=dim_hidden,
            name="seq_attend_embeddings_mapping")

        # fuse and convert
        fused_attends = tf.tanh(tf.add(dense_v_a, dense_h_a), name="fused_attends")
        fused_attends_shape = tf.shape(fused_attends)
        batch_size = fused_attends_shape[0]
        seq_length = fused_attends_shape[1]
        fused_attends = tf.reshape(fused_attends, shape=[-1, dim_hidden])

        # mapping backward from hidden space
        w_a = tf.Variable(tf.random_normal([1, dim_hidden]), dtype=data_type)
        attends = tf.matmul(w_a, fused_attends, transpose_b=True)
        attends = tf.reshape(attends, shape=(batch_size, seq_length, num_visual_features))

        # shape [batch, seq_max_length, num_visual]
        attend_weights = tf.nn.softmax(attends)
        attend_weights = tf.tile(tf.expand_dims(attend_weights, axis=-1),
                                 multiples=[1, 1, 1, tf.shape(seq_visual_embeddings)[-1]])

        # shape = [batch, seq_length, dim_visual_embedding]
        attend_visuals = tf.reduce_sum(tf.multiply(attend_weights, seq_visual_embeddings),
                                       axis=-2, name="attend_visual_embeddings")
        self.attend_visuals = attend_visuals

    @timeit
    @define_scope(scope_name="decoder")
    def __build_decoder(self):
        """
        decoder for caption outputs
        :return:
        """
        batch_size = self.batch_size
        data_type = self.model_config.data_type
        with tf.variable_scope("language",
                               initializer=self.initializer,
                               reuse=tf.AUTO_REUSE) as language_scope:
            language_rnn_cell = self.__create_rnn_cell()

            attend_initial_visuals = self.attend_visuals[:, 0, :]

            # language rnn_cell forward initial
            # shape = [batch, dim_visual_embedding + dim_attention]
            language_initial_inputs = tf.nn.l2_normalize(
                tf.concat([attend_initial_visuals, self.attend_initial_outputs], axis=-1),
                dim=-1, name="language_initial_inputs")
            language_zero_state = language_rnn_cell.zero_state(batch_size, data_type)

            _, language_initial_state = language_rnn_cell(
                language_initial_inputs, language_zero_state)

            language_inputs = tf.nn.l2_normalize(
                tf.concat([self.attend_visuals, self.attend_outputs], axis=-1),
                dim=-1, name="language_inputs")

        if self.mode == ModeKeys.INFER:
            self.language_initial_state = language_zero_state

            # In inference mode, states for feeding and fetching.
            self.language_state_feed = tf.placeholder(
                shape=[None, language_rnn_cell.state_size],
                name="language_state_feed", dtype=data_type)
            # Run one single language step.
            language_inputs = tf.squeeze(language_inputs, axis=1)
            language_outputs, self.language_new_state = language_rnn_cell(
                inputs=language_inputs, state=self.language_state_feed)

        else:
            language_outputs, language_state = tf.nn.dynamic_rnn(
                cell=language_rnn_cell,
                inputs=language_inputs,
                sequence_length=self.input_lengths,
                initial_state=language_zero_state,
                dtype=data_type)

        self.language_outputs = language_outputs

    @timeit
    @define_scope(scope_name="losses")
    def _build_loss(self):
        # Compute logits and weights
        vocab_size = self.data_reader.vocabulary.num_vocab
        self.outputs = self.language_outputs
        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=self.outputs,
                num_outputs=vocab_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)

        self.predicts = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        self.softmax = tf.nn.softmax(logits, name="softmax")
        if self.mode is not ModeKeys.INFER:
            weights = tf.sequence_mask(lengths=self.fw_target_lengths,
                                       dtype=self.outputs.dtype,
                                       name='masks')
            self.mask_weights = weights
            batch_loss = seq2seq.sequence_loss(
                logits=logits, targets=self.fw_target_seqs,
                weights=weights, name="sequence_loss")
            self.loss = batch_loss
            tf.summary.scalar("batch-loss", self.loss)

            correct_prediction = tf.equal(self.predicts, self.fw_target_seqs)
            batch_accuracy = tf.div(tf.reduce_sum(
                tf.multiply(tf.cast(correct_prediction, tf.float32), weights)),
                tf.reduce_sum(weights), name="batch_accuracy")
            self.accuracy = batch_accuracy
            tf.summary.scalar("accuracy", self.accuracy)
