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
            self.image_feature = tf.placeholder(tf.float32, [None, 4096], name='image_feature')
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

    @timeit
    @define_scope(scope_name='graph')
    def _build_graph(self):

        rnn_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.model_config.num_hidden_unit, state_is_tuple=True)
        if self.mode == "train":
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                rnn_cell,
                input_keep_prob=self.model_config.lstm_dropout_keep_prob,
                output_keep_prob=self.model_config.lstm_dropout_keep_prob)

        with tf.variable_scope("RNN", initializer=self.initializer) as rnn_scope:
            # Feed the image embeddings to set the initial LSTM state.
            zero_state = rnn_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            _, initial_state = rnn_cell(self.image_embeddings, zero_state)

        # Allow the RNN variables to be reused.
        rnn_scope.reuse_variables()

        if self.mode == ModeKeys.INFER:
            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
            self.initial_state = tf.concat(axis=1, values=initial_state, name="initial_state")
            # Placeholder for feeding a batch of concatenated states.
            self.state_feed = tf.placeholder(dtype=tf.float32,
                                        shape=[None, sum(rnn_cell.state_size)],
                                        name="state_feed")
            state_tuple = tf.split(value=self.state_feed, num_or_size_splits=2, axis=1)
            # Run a single step.
            inputs = tf.squeeze(self.input_seq_embeddings, axis=[1])
            outputs, state_tuple = rnn_cell(inputs=inputs, state=state_tuple)
            # Concatentate the resulting state.
            tf.concat(axis=1, values=state_tuple, name="state")

        else:
            # Run the batch of sequence embeddings through the LSTM.
            sequence_length = tf.reduce_sum(self.input_mask, 1)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                               inputs=self.input_seq_embeddings,
                                               sequence_length=sequence_length,
                                               initial_state=initial_state,
                                               dtype=tf.float32,
                                               scope=rnn_scope)
        # Stack batches vertically.
        # self.outputs = tf.reshape(outputs, [-1, rnn_cell.output_size])
        self.outputs = outputs

    @timeit
    def __create_rnn_cell(self, num_units):
        rnn_cell = GRUCell(num_units=num_units)
        if self.mode == ModeKeys.TRAIN:
            rnn_cell = DropoutWrapper(
                rnn_cell,
                input_keep_prob=self.model_config.dropout_keep_prob,
                output_keep_prob=self.model_config.dropout_keep_prob)
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
            # targets = tf.reshape(self.target_seqs, [-1])
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
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = batch_loss  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

            correct_prediction = tf.equal(self.predictions, self.target_seqs)
            batch_accuracy = tf.div(tf.reduce_sum(
                tf.multiply(tf.cast(correct_prediction, tf.float32), weights)),
                tf.reduce_sum(weights), name="batch_accuracy")
            self.accuracy = batch_accuracy
            tf.summary.scalar("accuracy", self.accuracy)
