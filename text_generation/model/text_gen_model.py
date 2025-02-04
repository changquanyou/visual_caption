# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf

from visual_caption.base.model.base_model import BaseModel


class TextGenerateModel(BaseModel):
    """
    Text generate model

    generate a text sentence for each descriptive phrase of region.

    """

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

    def _build_placeholder(self):
        print("......building placeholder begin......")
        self._input_sequence = tf.placeholder(name="input_sequence",
                                              shape=[self.config.batch_size, None],
                                              dtype=tf.int32)
        self._target_sequence = tf.placeholder(name="target_sequence",
                                               shape=[self.config.batch_size, None],
                                               dtype=tf.int32)
        self._build_embeddings()
        print("......building placeholder end......")

    def _build_embeddings(self):
        print("......building embeddings begin......")
        embedding_matrix = self._data_loader.token_embedding_matrix
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self._embeddings = tf.placeholder(dtype=tf.float32,
                                              shape=embedding_matrix.shape,
                                              name="_embeddings"
                                              )
            self._inputs = tf.nn.embedding_lookup(self._embeddings, self._input_sequence,
                                                  name="input_sequence_embeddings")

        print("......building embeddings end......")

    def __create_cell(self):
        cell = tf.contrib.rnn.GRUCell(self.config.hidden_neural_num, reuse=tf.get_variable_scope().reuse)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config.dropout_keep_prob)
        return cell

    def _build_network(self):
        print("......building deep network begin......")
        layer_num = self.config.hidden_layer_num
        data_type = self.config.data_type
        hidden_neural_num = self.config.hidden_neural_num
        tag_num = self._data_loader.tag_num

        """build the Bi_GRU network. Return the y_pred"""
        with tf.variable_scope("Bi_GRU") as scope_name:
            cell_fw = tf.contrib.rnn.MultiRNNCell([self.__create_cell() for _ in range(layer_num)], state_is_tuple=True)
            # initial_state_fw = cell_fw.zero_state(batch_size, data_type)
            cell_bw = tf.contrib.rnn.MultiRNNCell([self.__create_cell() for _ in range(layer_num)], state_is_tuple=True)
            # initial_state_bw = cell_bw.zero_state(batch_size, data_type)

            sequence_lengths = tf.reduce_sum(tf.sign(self._input_sequence + 1), 1)

            self._sequence_lengths = sequence_lengths
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self._inputs,
                sequence_length=sequence_lengths,
                # initial_state_fw=initial_state_fw,
                # initial_state_bw=initial_state_bw,
                dtype=data_type
            )
        # outputs is a length T list of output vectors, which is [batch_size, 2 * hidden_size]
        # [time][batch][cell_fw.output_size + cell_bw.output_size]

        self._outputs = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
        # output has size: [T, size * 2]
        print("......building networks finished......")

    def _build_loss(self):
        print("......building loss......")
        # Compute logits and weights
        hidden_size = self.config.hidden_neural_num
        vocab_size = self._data_loader.vocab_size
        data_type = self.config.data_type
        batch_size = self.config.batch_size

        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", [hidden_size * 2, vocab_size], dtype=data_type)
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type)

        with tf.variable_scope("logits"):
            logits = tf.matmul(self._outputs, softmax_w) + softmax_b  # logits shape[time_step, target_num]

        # Computing losses.
        with tf.variable_scope("loss"):
            # adding extra statistics to monitor
            targets = self._target_seqs
            correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(targets, [-1]))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]), logits=logits)
            self._cost = tf.reduce_mean(loss)  # loss
            tf.summary.scalar("accuracy", self._accuracy)
            tf.summary.scalar("loss", self._cost)
        print("......building loss finished......")
