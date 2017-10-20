# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import sys
import time

import tensorflow as tf

from visual_caption.base.model.base_model import BaseModel, timeit, define_scope
from visual_caption.image_caption.data.data_embedding import ImageCaptionDataEmbedding


class ImageCaptionModel(BaseModel):
    def __init__(self, config, data_reader):
        super().__init__(config=config, data_reader=data_reader)

    @timeit
    @define_scope(scope_name='embeddings')
    def _build_embeddings(self):

        self._data_embedding = ImageCaptionDataEmbedding()
        self._embeddings = tf.Variable(self._data_embedding.token_embedding_matrix,
                                       dtype=self.config.data_type,
                                       trainable=self.config.train_embeddings,
                                       name='token_embeddings')
        pass

    @timeit
    @define_scope(scope_name='inputs')
    def _build_inputs(self):
        # input images and seqs batch
        batch_size = self.data_reader.data_config.batch_size
        data_config = self.data_reader.data_config

        train_inputs = self.data_reader._build_data_inputs(data_config.train_data_dir)
        # test_inputs = self._data_reader._build_data_inputs(data_config.test_data_dir)
        validation_inputs = self.data_reader._build_data_inputs(data_config.validation_data_dir)

        self.input_image_embeddings = tf.placeholder(dtype=tf.float32,
                                                     shape=[batch_size, 4096],
                                                     name='image_embeddings')
        if self.config.mode == 'train':
            images_batch, input_seqs_batch, target_seqs_batch, input_mask_batch = train_inputs
        # elif self.config.mode == 'test':
        #     images_batch, input_seqs_batch, target_seqs_batch, input_mask_batch = test_inputs
        elif self.config.mode == 'validation':
            images_batch, input_seqs_batch, target_seqs_batch, input_mask_batch = validation_inputs

        self.input_image_embeddings = images_batch
        self.input_seq_embeddings = tf.nn.embedding_lookup(params=self._embeddings,
                                                           ids=input_seqs_batch,
                                                           name="input_seqs_embeddings")
        self.target_seqs = target_seqs_batch
        self.input_masks = input_mask_batch

    def __create_rnn_cell(self):
        # This RNN cell has biases and outputs tanh(new_c) * sigmoid(o), but the
        # modified RNN in the "Show and Tell" paper has no biases and outputs
        # new_c * sigmoid(o).
        rnn_cell = tf.contrib.rnn.GRUCell(num_units=self.config.hidden_neural_num)
        if self.config.mode == "train":
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell,
                                                     input_keep_prob=self.config.dropout_keep_prob,
                                                     output_keep_prob=self.config.dropout_keep_prob)
        return rnn_cell

    @timeit
    @define_scope(scope_name='network')
    def _build_network(self):
        layer_num = self.config.hidden_layer_num
        data_type = self.config.data_type
        hidden_neural_num = self.config.hidden_neural_num

        rnn_cell = self.__create_rnn_cell()
        # Feed the image embeddings to set the initial LSTM state.
        zero_state = rnn_cell.zero_state(
            batch_size=self.input_image_embeddings.get_shape()[0], dtype=tf.float32)
        _, initial_state = rnn_cell(self.input_image_embeddings, zero_state)

        # stack multi layers RNN
        # cells_forward = tf.contrib.rnn.MultiRNNCell(cells=[self.__create_rnn_cell() for _ in range(layer_num)],
        #                                             state_is_tuple=True)
        # cells_backward = tf.contrib.rnn.MultiRNNCell(cells=[self.__create_rnn_cell() for _ in range(layer_num)],
        #                                              state_is_tuple=True)

        if self.config.mode == "inference":
            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
            tf.concat(axis=1, values=initial_state, name="initial_state")
            # Placeholder for feeding a batch of concatenated states.
            state_feed = tf.placeholder(dtype=tf.float32,
                                        shape=[None, sum(rnn_cell.state_size)],
                                        name="state_feed")
            state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)
            # Run a single run step.
            outputs, state_tuple = rnn_cell(inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
                                            state=state_tuple)
            # Concatentate the resulting state.
            tf.concat(axis=1, values=state_tuple, name="state")
        else:
            sequence_lengths = tf.reduce_sum(self.input_masks, 1)
            cell_fw = self.__create_rnn_cell()
            cell_bw = self.__create_rnn_cell()
            # cell_fw = cells_forward
            # cell_bw = cells_backward

            outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.input_seq_embeddings,
                sequence_length=sequence_lengths,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                dtype=data_type
            )
        # outputs is a length T list of output vectors, which is [batch_size, 2 * hidden_size]
        # [time][batch][cell_fw.output_size + cell_bw.output_size]
        self._outputs = tf.reshape(tf.concat(outputs, 1), [-1, hidden_neural_num * 2])
        # output has size: [T, size * 2]

        # Compute logits and weights
        hidden_size = self.config.hidden_neural_num
        vocab_size = self._data_embedding.vocab_size
        data_type = self.config.data_type

        with tf.variable_scope('logits'):
            softmax_w = tf.get_variable("softmax_w", [hidden_size * 2, vocab_size], dtype=data_type)
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type)
            self.logits = tf.matmul(self._outputs, softmax_w) + softmax_b  # logits shape[time_step, target_num]


    @timeit
    @define_scope(scope_name='losses')
    def _build_loss(self):
        # adding extra statistics to monitor
        targets = self.target_seqs
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int64), tf.reshape(targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar("accuracy", self.accuracy)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]),
                                                              logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", self.loss)
        pass


class ImageCaptionAttentionModel(BaseModel):
    """
    Image Caption Model build with BiRNN and Attention
    """

    def _build_inputs(self):
        """
        build inputs placeholders for computing graph
        :return:
        """
