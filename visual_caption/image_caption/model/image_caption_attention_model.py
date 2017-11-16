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
    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionAttentionModel, self).__init__(model_config, data_reader, mode)

    @timeit
    @define_scope(scope_name="inputs")
    def _build_inputs(self):
        data_type = self.model_config.data_type

        self.image_ids = tf.placeholder(shape=[None],
                                        name='image_ids',
                                        dtype=tf.string)

        self.input_images = tf.placeholder(shape=[None, 299, 299, 3],
                                           name='input_images',
                                           dtype=data_type)

        self.input_regions = tf.placeholder(shape=[None, None, 299, 299, 3],
                                            name='input_regions',
                                            dtype=data_type)

        self.input_seqs = tf.placeholder(shape=[None, None],
                                         name='input_seqs',
                                         dtype=tf.int32)

        self.target_seqs = tf.placeholder(shape=[None, None],
                                          name='target_seqs',
                                          dtype=tf.int32)

        self.input_lengths = tf.placeholder(shape=[None],
                                            name='input_lengths',
                                            dtype=tf.int32)

        self.target_lengths = tf.placeholder(shape=[None],
                                             name='target_lengths',
                                             dtype=tf.int32)

        # replace default model config batch_size with data pipeline batch_size
        self.batch_size = tf.shape(self.input_images)[0]
        # Maximum decoder time_steps in current batch
        self.max_seq_length = self.model_config.length_max_output

        pass

    @timeit
    @define_scope(scope_name="embeddings")
    def _build_embeddings(self):
        vocab_file = self.model_config.data_config.vocab_file
        embedding_size = self.model_config.data_config.embedding_size
        # Save the embedding size in the graph.
        tf.constant(embedding_size, name="embedding_size")

        self.vocab_table = lookup_ops.index_table_from_file(vocabulary_file=vocab_file)
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.seq_embedding_map = tf.get_variable(
                shape=[self.vocab_table.size(), self.model_config.embedding_size],
                dtype=self.model_config.data_type,
                initializer=self.emb_initializer,
                trainable=self.model_config.train_embeddings,
                name='seq_embedding_map')
            seq_embeddings = tf.nn.embedding_lookup(self.seq_embedding_map, self.input_seqs)
        self.input_seq_embeddings = seq_embeddings

        # Map image feature outputs into embedding space.
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
        self.__build_decoder()
        pass

    @timeit
    @define_scope(scope_name="encoder")
    def __build_encoder(self):
        pass

    @timeit
    def __get_input_embeddings(self, seq_embeddings, image_embeddings):

        image_embedding_seqs = tf.expand_dims(input=image_embeddings, axis=1)
        image_embedding_seqs = tf.tile(input=image_embedding_seqs,
                                       multiples=[1, tf.shape(seq_embeddings)[1], 1])
        input_embeddings = tf.concat(values=[self.input_seq_embeddings, image_embedding_seqs],
                                     axis=-1)
        input_embeddings = tf.nn.l2_normalize(input_embeddings, dim=-1)
        return input_embeddings
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
    @define_scope(scope_name="decoder")
    def __build_decoder(self):

        # put token begin as start state
        token2index = self._data_embedding.token2index
        token_begin = self.model_config.data_config.token_begin

        token_begin_ids = tf.fill([self.batch_size], token2index[token_begin])
        token_begin_embeddings = tf.nn.embedding_lookup(params=self.seq_embedding_map,
                                                        ids=token_begin_ids)
        start_embeddings = self.__get_input_embeddings(
            seq_embeddings=token_begin_embeddings,
            image_embeddings=self.image_embeddings
        )
        # create inputs for rnn_cell
        inputs = self.__get_input_embeddings(
            seq_embeddings=self.input_seq_embeddings,
            image_embeddings=self.image_embeddings)
        with tf.variable_scope("rnn_decoder", initializer=self.initializer,
                               reuse=tf.AUTO_REUSE) as rnn_scope:
            # forward RNN cell
            cell_fw = self.__create_rnn_cell()
            # backward RNN cell
            cell_bw = self.__create_rnn_cell()

            # Feed the image embeddings to set the initial rnn_cell state.
            zero_state_fw = cell_fw.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            _, initial_state_fw = cell_fw(start_embeddings, zero_state_fw)
            zero_state_bw = cell_bw.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            _, initial_state_bw = cell_bw(start_embeddings, zero_state_bw)

        if self.mode == ModeKeys.INFER:
            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
            self.initial_states = tf.concat(tf.concat([initial_state_fw, initial_state_bw], axis=-1),
                                            axis=1, name="initial_state")
            # Placeholder for feeding a batch of concatenated states.
            self.state_feeds = tf.placeholder(shape=[None, cell_fw.state_size + cell_bw.state_size],
                                              name="state_feed", dtype=tf.float32)
            state_tuple_fw, state_tuple_bw = tf.split(value=self.state_feeds, num_or_size_splits=2, axis=1)

            input_embeddings = self.__get_input_embeddings(
                seq_embeddings=token_begin_embeddings,
                image_embeddings=self.image_embeddings
            )

            inputs_fw = tf.squeeze(input_embeddings, axis=[1], name="input_fw")
            inputs_bw = tf.squeeze(input_embeddings, axis=[1], name="input_bw")

            # Run a single step.
            outputs_fw, state_tuple_new_fw = cell_fw(inputs=inputs_fw, state=state_tuple_fw)
            outputs_bw, state_tuple_new_bw = cell_bw(inputs=inputs_bw, state=state_tuple_bw)

            outputs = (outputs_fw, outputs_bw)
            final_states = (state_tuple_new_fw, state_tuple_new_bw)
        else:
            data_type = self.model_config.data_type
            # Run the batch sequences through the rnn_cell.
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                sequence_length=self.input_lengths,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=data_type,
                scope=rnn_scope)

        self.outputs = tf.concat(outputs, -1)
        self.final_states = tf.concat(final_states, -1)
        pass

    @timeit
    @define_scope(scope_name="losses")
    def _build_loss(self):
        # Compute logits and weights
        vocab_size = self.vocab_table.size()
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
            batch_loss = seq2seq.sequence_loss(logits=logits,
                                               targets=self.target_seqs,
                                               weights=weights)
            self.loss = batch_loss
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()
            # Add summaries.
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            self.total_loss = total_loss
            self.target_cross_entropy_losses = batch_loss  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

            correct_prediction = tf.equal(self.predictions, self.target_seqs)
            batch_accuracy = tf.div(tf.reduce_sum(
                tf.multiply(tf.cast(correct_prediction, tf.float32), weights)),
                tf.reduce_sum(weights), name="batch_accuracy")
            self.accuracy = batch_accuracy
            tf.summary.scalar("accuracy", self.accuracy)

    @timeit
    def _decode(self):
        assert self.mode == ModeKeys.INFER

    @timeit
    @define_scope(scope_name="attention")
    def _build_attention(self):
        attention_cell = self.__create_rnn_cell()
        pass
