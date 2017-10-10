# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf

from visual_caption.base.model.base_model import BaseModel


class ImageCaptionModel(BaseModel):
    def __init__(self, config, data_reader):
        super().__init__(config=config, data_reader=data_reader)
        self.batch_size = self._data_reader.data_config.batch_size

    def _build_inputs(self):
        print("......building inputs begin......")
        # input images and seqs batch

        self.input_images, self.input_seqs, self.target_seqs, self.input_mask = self._data_reader.build_data_inputs()

        self._embeddings = tf.Variable(self._data_reader.token_embedding_matrix,
                                       dtype=self.config.data_type,
                                       trainable=self.config.train_embeddings,
                                       name='token_embedding')

        self.input_image_embeddings = self.input_images

        self.input_seq_embeddings = tf.nn.embedding_lookup(params=self._embeddings,
                                                           ids=self.input_seqs,
                                                           name="input_seqs_embeddings")
        self._input_mask = self.input_mask

        self._inputs = (self.input_image_embeddings, self.input_seq_embeddings, self.target_seqs, self.input_mask)

        print("......building inputs end......")

    def __create_cell(self):
        cell = tf.contrib.rnn.GRUCell(self.config.hidden_neural_num, reuse=tf.get_variable_scope().reuse)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config.dropout_keep_prob)
        return cell

    def _build_network(self):
        print("......building network begin......")

        layer_num = self.config.hidden_layer_num
        data_type = self.config.data_type
        hidden_neural_num = self.config.hidden_neural_num

        """build the Bi_GRU network. Return the y_pred"""
        with tf.variable_scope("Bi_GRU") as scope_name:
            cell_fw = tf.contrib.rnn.MultiRNNCell([self.__create_cell() for _ in range(layer_num)], state_is_tuple=True)
            # initial_state_fw = cell_fw.zero_state(batch_size, data_type)
            cell_bw = tf.contrib.rnn.MultiRNNCell([self.__create_cell() for _ in range(layer_num)], state_is_tuple=True)
            # initial_state_bw = cell_bw.zero_state(batch_size, data_type)

            sequence_lengths = tf.reduce_sum(tf.sign(self.input_seqs + 1), 1)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.input_seq_embeddings,
                sequence_length=sequence_lengths,
                # initial_state_fw=initial_state_fw,
                # initial_state_bw=initial_state_bw,
                dtype=data_type
            )
        # outputs is a length T list of output vectors, which is [batch_size, 2 * hidden_size]
        # [time][batch][cell_fw.output_size + cell_bw.output_size]
        self._outputs = tf.reshape(tf.concat(outputs, 1), [-1, hidden_neural_num * 2])
        # output has size: [T, size * 2]
        print("......building network end......")

    def _build_fetches(self):
        print("......building fetches begin......")
        self._fetches = [self._train_op, self._cost, self._merged]
        print("......building fetches end......")

    def _build_loss(self):
        print("......building loss begin......")
        # Compute logits and weights
        hidden_size = self.config.hidden_neural_num
        vocab_size = self._data_reader.vocab_size
        data_type = self.config.data_type

        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable("softmax_w", [hidden_size * 2, vocab_size], dtype=data_type)
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type)

        with tf.variable_scope("logits"):
            logits = tf.matmul(self._outputs, softmax_w) + softmax_b  # logits shape[time_step, target_num]

        # Computing losses.
        with tf.variable_scope("loss"):
            # adding extra statistics to monitor
            targets = self.target_seqs
            correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int64), tf.reshape(targets, [-1]))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]), logits=logits)
            self._cost = tf.reduce_mean(loss)  # loss
            tf.summary.scalar("accuracy", self._accuracy)
            tf.summary.scalar("loss", self._cost)
        print("......building loss end......")
        pass

