# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.ops.rnn_cell_impl import GRUCell, DropoutWrapper

from visual_caption.base.model.base_model import BaseModel
from visual_caption.image_caption.data.data_embedding import ImageCaptionDataEmbedding
from visual_caption.utils.decorator_utils import timeit, define_scope


class ImageRNNModel(BaseModel):
    def __init__(self, model_config, data_reader, mode):
        super(ImageRNNModel, self).__init__(
            model_config=model_config,
            data_reader=data_reader,
            mode=mode
        )

    @timeit
    @define_scope(scope_name='inputs')
    def _build_inputs(self):
        data_type = self.model_config.data_type
        visual_feature_size = self.model_config.data_config.visual_feature_size

        # A float32 Tensor with shape [1]
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')

        if self.mode == ModeKeys.INFER:
            # In inference mode, images and inputs are fed via placeholders.
            # image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            self.input_feed = tf.placeholder(dtype=tf.int64,
                                             shape=[None],  # batch_size
                                             name="input_feed")
            input_seqs = tf.expand_dims(self.input_feed, 1)
            # A float32 Tensor with shape [batch_size, image_feature_size].
            self.image_feature = tf.placeholder(dtype=tf.float32,
                                                shape=[None, visual_feature_size],
                                                name='image_feature')
            self.input_seqs = input_seqs

        else:

            (image_ids, image_features, captions, targets,
             caption_ids, target_ids, caption_lengths, target_lengths) = self.next_batch

            self.image_feature = image_features
            self.image_ids = image_ids

            self.input_seqs = caption_ids
            self.target_seqs = target_ids

            self.input_lengths = caption_lengths
            self.target_lengths = target_lengths

        # replace default model config batch_size with data pipeline batch_size
        self.batch_size = tf.shape(self.image_feature)[0]
        # Maximum decoder time_steps in current batch
        self.max_seq_length = self.model_config.length_max_output

        pass

    @timeit
    @define_scope(scope_name='embeddings')
    def _build_embeddings(self):
        self._data_embedding = ImageCaptionDataEmbedding()
        embedding_size = self._data_embedding.embedding_size

        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.embedding_map = tf.Variable(self._data_embedding.token_embedding_matrix,
                                             dtype=self.model_config.data_type,
                                             trainable=self.model_config.train_embeddings,
                                             name='embedding_map')
        self.input_seq_embeddings = tf.nn.embedding_lookup(
            params=self.embedding_map, ids=self.input_seqs)

        # Map inception output into embedding space.
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=self.image_feature,
                num_outputs=embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

        # Save the embedding size in the graph.
        tf.constant(embedding_size, name="embedding_size")
        self.image_embeddings = image_embeddings

        pass

    def _create_rnn_cell(self, num_units):
        rnn_cell = GRUCell(num_units)
        if self.mode == ModeKeys.TRAIN:
            rnn_cell = DropoutWrapper(cell=rnn_cell,
                                      input_keep_prob=self.model_config.dropout_keep_prob,
                                      output_keep_prob=self.model_config.dropout_keep_prob)
        return rnn_cell

    @timeit
    @define_scope(scope_name='graph')
    def _build_graph(self):
        num_units = self.model_config.num_hidden_unit

        # put token begin as start state
        token2index = self._data_embedding.token2index
        token_begin = self.model_config.data_config.token_begin
        token_begin_ids = tf.fill([self.batch_size], token2index[token_begin])
        token_begin_embeddings = tf.nn.embedding_lookup(params=self.embedding_map, ids=token_begin_ids)

        start_embeddings = tf.concat(values=[token_begin_embeddings, self.image_embeddings], axis=-1)
        start_embeddings = tf.nn.l2_normalize(start_embeddings, dim=-1)

        with tf.variable_scope("RNN", initializer=self.initializer, reuse=tf.AUTO_REUSE) as rnn_scope:
            # Feed the image embeddings to set the initial RNN state.
            # forward RNN cell
            cell_fw = self._create_rnn_cell(num_units=num_units)
            # backward RNN cell
            cell_bw = self._create_rnn_cell(num_units=num_units)

            zero_state_fw = cell_fw.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            _, initial_state_fw = cell_fw(start_embeddings, zero_state_fw)

            zero_state_bw = cell_bw.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            _, initial_state_bw = cell_bw(start_embeddings, zero_state_bw)

        if self.mode == ModeKeys.INFER:
            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
            self.initial_state = tf.concat(tf.concat([initial_state_fw, initial_state_bw], axis=-1),
                                           axis=1, name="initial_state")
            # Placeholder for feeding a batch of concatenated states.
            self.state_feed = tf.placeholder(shape=[None, cell_fw.state_size + cell_bw.state_size],
                                             name="state_feed", dtype=tf.float32)

            state_tuple_fw, state_tuple_bw = tf.split(value=self.state_feed, num_or_size_splits=2, axis=1)

            # Run a single step.
            inputs_fw = tf.squeeze(self.input_seq_embeddings, axis=[1], name="input_fw")
            inputs_bw = tf.squeeze(self.input_seq_embeddings, axis=[1], name="input_bw")

            outputs_fw, state_tuple_new_fw = cell_fw(inputs=inputs_fw, state=state_tuple_fw)
            outputs_bw, state_tuple_new_bw = cell_bw(inputs=inputs_bw, state=state_tuple_bw)

            outputs = (outputs_fw, outputs_bw)

            # Concatentate the resulting state.
            tf.concat(axis=1, values=state_tuple_new_fw, name="state_fw")
            tf.concat(axis=1, values=state_tuple_new_bw, name="state_bw")

        else:
            # Run the batch of sequence embeddings through the LSTM.
            image_embedding_seqs = tf.expand_dims(input=self.image_embeddings, axis=1)
            image_embedding_seqs = tf.tile(image_embedding_seqs,
                                           multiples=[1, tf.reduce_max(self.input_lengths), 1])
            input_embeddings = tf.concat(values=[self.input_seq_embeddings, image_embedding_seqs],
                                         axis=-1)
            input_embeddings = tf.nn.l2_normalize(input_embeddings, dim=-1)
            # input_embeddings = self.input_seq_embeddings
            outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=input_embeddings,
                sequence_length=self.input_lengths,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32,
                scope=rnn_scope)
        # Stack batches vertically.
        # self.outputs = tf.reshape(outputs, [-1, rnn_cell.output_size])
        self.outputs = tf.concat(outputs, -1)

    @timeit
    def __create_rnn_cell(self, num_units):
        rnn_cell = GRUCell(num_units=num_units)
        keep_prob = self.model_config.dropout_keep_prob
        if self.mode == ModeKeys.TRAIN:
            rnn_cell = DropoutWrapper(cell=rnn_cell,
                                      input_keep_prob=keep_prob,
                                      output_keep_prob=keep_prob)
        return rnn_cell

    @timeit
    @define_scope(scope_name="losses")
    def _build_loss(self):
        # Compute logits and weights
        vocab_size = self._data_embedding.vocab_size
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
                                       maxlen=tf.reduce_max(self.target_lengths),
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
