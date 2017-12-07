# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.ops import lookup_ops

from visual_caption.base.model.base_model import BaseModel
from visual_caption.image_caption.model import model_helper
from visual_caption.utils.decorator_utils import timeit, define_scope


class ImageCaptionAttentionModel(BaseModel):
    """
        Image Caption Model with the below mechanism:
            1.Bottom-Up and Top-Down Attention
            2.Multi-modal Factorized Bilinear Pooling with Co-Attention

    """

    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionAttentionModel, self).__init__(model_config, data_reader, mode)

    @timeit
    @define_scope(scope_name="inputs")
    def _build_inputs(self):
        data_type = self.model_config.data_type

        self.image_ids = tf.placeholder(shape=[None],
                                        name='image_ids',
                                        dtype=tf.string)

        # image features, shape=[batch, image_feature]
        self.input_images = tf.placeholder(shape=[None, None],
                                           name='input_images',
                                           dtype=data_type)

        # features of regions, shape=[batch, region_order, region_feature]
        self.input_regions = tf.placeholder(shape=[None, None, None],
                                            name='input_regions',
                                            dtype=data_type)

        # batched input sequences shape=[batch, sequence_ids]
        self.input_seqs = tf.placeholder(shape=[None, None],
                                         name='input_seqs',
                                         dtype=tf.int32)

        self.input_lengths = tf.placeholder(shape=[None],
                                            name='input_lengths',
                                            dtype=tf.int32)

        # batched target sequences shape=[batch, sequence_ids]
        self.target_seqs = tf.placeholder(shape=[None, None],
                                          name='target_seqs',

                                          dtype=tf.int32)

        self.target_lengths = tf.placeholder(shape=[None],
                                             name='target_lengths',
                                             dtype=tf.int32)

        # replace default model_config batch_size with data pipeline batch_size
        self.batch_size = tf.shape(self.input_images)[0]

        # Maximum decoder time_steps in current batch
        self.max_seq_length = self.model_config.length_max_output

        pass

    @timeit
    @define_scope(scope_name="embeddings")
    def _build_embeddings(self):
        vocab_file = self.model_config.data_config.vocab_file
        embedding_size = self.model_config.data_config.dim_token_feature

        vocab_num = len(self.data_reader.index2token)

        # Save the embedding size in the graph.
        tf.constant(embedding_size, name="embedding_size")
        self.vocab_table = lookup_ops.index_table_from_file(vocabulary_file=vocab_file)
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.seq_embedding_map = tf.get_variable(
                shape=[vocab_num, embedding_size],
                dtype=self.model_config.data_type,
                initializer=self.emb_initializer,
                trainable=self.model_config.train_embeddings,
                name='seq_embedding_map')
            seq_embeddings = tf.nn.embedding_lookup(self.seq_embedding_map, self.input_seqs)
        self.input_seq_embeddings = seq_embeddings

        # for token begin batch embeddings
        start_embedding = tf.nn.embedding_lookup(self.seq_embedding_map, self.token_start_id)
        self.start_seq_embeddings = tf.tile(input=start_embedding,
                                            multiples=[self.batch_size, 1])

        # for token end batch embeddings
        token_end = self.model_config.data_config.token_end
        end_embedding = tf.nn.embedding_lookup(self.seq_embedding_map, token_end)
        self.end_seq_embeddings = tf.tile(input=end_embedding,
                                          multiples=[self.batch_size, 1])

        # Mapping image and region features into embedding space.
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=self.input_images,
                num_outputs=embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)
        self.image_embeddings = image_embeddings
        pass

    @timeit
    @define_scope(scope_name="graph")
    def _build_graph(self):
        self.__build_encoder()
        self.__build_attention()
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
        dropout = self.model_config.dropout
        forget_bias = self.model_config.forget_bias
        unit_type = self.model_config.unit_type
        num_units = self.model_config.num_hidden_unit
        num_layers = self.model_config.num_layers
        num_residual_layers = self.model_config.num_residual_layers
        num_gpus = self.model_config.num_gpus
        mode = self.mode
        rnn_cell = model_helper.create_rnn_cell(
            unit_type=unit_type, num_units=num_units, num_layers=num_layers,
            dropout=dropout, forget_bias=forget_bias, num_residual_layers=num_residual_layers,
            mode=mode, num_gpus=num_gpus)
        return rnn_cell
        pass

    @timeit
    def __build_attention(self):
        """
        build attention model
        :return:
        """
        batch_size = self.batch_size
        data_type = self.model_config.data_type
        max_input_length = tf.reduce_max(self.input_lengths, axis=-1)
        with tf.variable_scope("attention",
                               initializer=self.initializer,
                               reuse=tf.AUTO_REUSE) as attend_scope:
            attend_fw_rnn_cell = self.__create_rnn_cell()
            attend_bw_rnn_cell = self.__create_rnn_cell()

            # attend rnn_cell initial, shape=[batch,dim_visual_embedding]
            average_visuals = tf.reduce_mean(self.input_visual_embeddings,
                                             axis=1, name="average_visuals")

            # initial fw token attend input
            attend_fw_initial_inputs = tf.nn.l2_normalize(
                tf.concat([average_visuals, self.start_seq_embeddings], axis=-1),
                name="attend_fw_initial_inputs", dim=-1)
            attend_fw_zero_state = attend_fw_rnn_cell.zero_state(
                batch_size=batch_size, dtype=data_type)
            attend_fw_initial_outputs, attend_fw_initial_state = attend_fw_rnn_cell(
                attend_fw_initial_inputs, attend_fw_zero_state)

            # initial bw token attend input
            attend_bw_initial_inputs = tf.nn.l2_normalize(
                tf.concat([average_visuals, self.end_seq_embeddings], axis=-1),
                name="attend_bw_initial_inputs", dim=-1)
            attend_bw_zero_state = attend_bw_rnn_cell.zero_state(
                batch_size=batch_size, dtype=data_type)
            attend_bw_initial_outputs, attend_bw_initial_state = attend_bw_rnn_cell(
                attend_bw_initial_inputs, attend_bw_zero_state)

            # for attend rnn train and eval
            # [batch,seq_max_length,dim_token_embedding]
            seq_average_visuals = tf.tile(tf.expand_dims(average_visuals, axis=1),
                                          multiples=[1, max_input_length, 1])
            attend_inputs = tf.nn.l2_normalize(
                tf.concat([seq_average_visuals, self.input_seq_embeddings], axis=-1),
                name="attend_inputs", dim=-1)

            attend_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=attend_fw_rnn_cell,
                cell_bw=attend_bw_rnn_cell,
                inputs=attend_inputs,
                sequence_length=self.input_lengths,
                initial_state_fw=attend_fw_initial_state,
                initial_state_bw=attend_bw_initial_state,
                dtype=data_type)

            attend_outputs = tf.add_n(attend_outputs, name="attend_outputs")

        self.attend_fw_initial_outputs = attend_fw_initial_outputs
        self.attend_bw_initial_outputs = attend_bw_initial_outputs
        self.attend_outputs = attend_outputs
        pass

    @timeit
    @define_scope(scope_name="attend")
    def __attend(self):
        """
         to compute the attend visual features according attend_outputs
        :param attend_outputs:
        :return:
        """
        data_type = self.model_config.data_type
        num_visual_features = self.data_config.num_visual_features
        max_input_length = tf.reduce_max(self.input_lengths, axis=-1)
        dim_hidden = 1000

        # expand to sequence length,
        # shape = [batch, seq_length, num_regions, dim_visual_embedding]
        seq_visual_embeddings = tf.tile(
            tf.expand_dims(self.input_visual_embeddings, axis=1),
            multiples=[1, max_input_length, 1, 1],
            name="seq_visual_embeddings"
        )
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
        # num_regions = fused_attends_shape[2]

        fused_attends = tf.reshape(fused_attends, shape=[-1, dim_hidden])
        # mapping backward from hidden space
        w_a = tf.Variable(tf.random_normal([1, dim_hidden]), dtype=data_type)
        attends = tf.matmul(w_a, fused_attends, transpose_b=True)
        attends = tf.reshape(attends, shape=(batch_size, seq_length, num_visual_features))
        # shape [batch, seq_max_length, num_visual]
        attend_weights = tf.nn.softmax(attends)
        attend_weights = tf.tile(tf.expand_dims(attend_weights, axis=-1),
                                 multiples=[1, 1, 1, tf.shape(seq_visual_embeddings)[-1]])

        # shape = [batch, seq_max_length, dim_visual_embedding]
        attend_visuals = tf.reduce_sum(
            tf.multiply(attend_weights, seq_visual_embeddings),
            axis=-2, name="attend_visual_embeddings")

        # batch start and end token visual initial
        attend_fw_initial_visuals = attend_visuals[:, 0, :]
        attend_bw_initial_visuals = attend_visuals[:, -1, :]

        self.attend_fw_initial_visuals = attend_fw_initial_visuals
        self.attend_bw_initial_visuals = attend_bw_initial_visuals
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
            language_rnn_fw_cell = self.__create_rnn_cell()
            language_rnn_bw_cell = self.__create_rnn_cell()

            # language rnn_cell forward initial
            # shape = [batch, dim_visual_embedding + dim_attention]
            language_fw_initial_inputs = tf.nn.l2_normalize(
                tf.concat([self.attend_fw_initial_visuals, self.attend_fw_initial_outputs], axis=-1),
                dim=-1, name="language_fw_initial_inputs")
            language_fw_zero_state = language_rnn_fw_cell.zero_state(batch_size, data_type)
            _, language_fw_initial_state = language_rnn_fw_cell(
                language_fw_initial_inputs, language_fw_zero_state)

            # language rnn_cell backward initial
            # shape = [batch, dim_visual_embedding + dim_attention]
            language_bw_initial_inputs = tf.nn.l2_normalize(
                tf.concat([self.attend_bw_initial_visuals, self.attend_bw_initial_outputs], axis=-1),
                dim=-1, name="language_bw_initial_inputs")
            language_bw_zero_state = language_rnn_bw_cell.zero_state(batch_size, data_type)
            _, language_bw_initial_state = language_rnn_bw_cell(
                language_bw_initial_inputs, language_bw_zero_state)

            #  train and eval inputs for language_rnn_cell
            #  shape=[batch, seq_lengths,[visual_embedding+attend_size]]
            language_inputs = tf.nn.l2_normalize(
                tf.concat([self.attend_visuals, self.attend_outputs], axis=-1),
                dim=-1, name="language_inputs")

            language_outputs, language_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=language_rnn_fw_cell,
                cell_bw=language_rnn_bw_cell,
                inputs=language_inputs,
                sequence_length=self.input_lengths,
                initial_state_fw=language_fw_initial_state,
                initial_state_bw=language_bw_initial_state,
                dtype=data_type)

        self.decoder_outputs = tf.concat(values=language_outputs,
                                         axis=-1, name="language_outputs")


    @timeit
    def _build_loss(self):
        # Compute logits and weights
        vocab_size = self.vocab_table.size()

        self.outputs = self.decoder_outputs
        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=self.outputs,
                num_outputs=vocab_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)

        self.predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        if self.mode == ModeKeys.INFER:
            self.softmax = tf.nn.softmax(logits, name="softmax")
        else:
            weights = tf.sequence_mask(lengths=self.target_lengths,
                                       dtype=self.outputs.dtype,
                                       name='masks')
            with tf.variable_scope("losses"):
                batch_loss = seq2seq.sequence_loss(
                    logits=logits,
                    targets=self.target_seqs,
                    weights=weights)
                self.loss = batch_loss
                tf.losses.add_loss(batch_loss)
                total_loss = tf.losses.get_total_loss()
                # Add summaries.
                tf.summary.scalar("batch_loss", batch_loss)
                tf.summary.scalar("total_loss", total_loss)
                self.total_loss = total_loss
                self.target_cross_entropy_losses = batch_loss  # Used in evaluation.
                self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

            with tf.variable_scope("accuracy"):
                correct_prediction = tf.equal(self.predictions, self.target_seqs)
                batch_accuracy = tf.div(tf.reduce_sum(
                    tf.multiply(tf.cast(correct_prediction, tf.float32), weights)),
                    tf.reduce_sum(weights), name="batch_accuracy")
                self.accuracy = batch_accuracy
                tf.summary.scalar("accuracy", self.accuracy)

    @timeit
    def _decode(self):

        """
        decode:
            p(y(t)|y(1:t-1)) = softmax( W(p)* h_language(t)+b(p) )
        :return:
        """

        assert self.mode == ModeKeys.INFER
