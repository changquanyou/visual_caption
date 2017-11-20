# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding
import sys
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper

from slim.nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
sys.append('../../../')
from visual_caption.image_caption.data.data_embedding import ImageCaptionDataEmbedding
from visual_caption.image_caption.model.image_rnn_model import ImageRNNModel
from visual_caption.utils.decorator_utils import timeit, define_scope


class FasterRCNNModel(ImageRNNModel):
    def __init__(self, model_config, data_reader, mode):
        super(FasterRCNNModel, self).__init__(
            model_config=model_config, data_reader=data_reader, mode=mode)
        pass

    @timeit
    @define_scope(scope_name='inputs')
    def _build_inputs(self):
        data_type = self.model_config.data_type

        self.image_ids = tf.placeholder(shape=[None],
                                        name='image_ids',
                                        dtype=tf.string)
        self.input_images = tf.placeholder(shape=[None, 299, 299, 3],
                                           name='input_images',
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

    @timeit
    @define_scope(scope_name='embeddings')
    def _build_embeddings(self):
        self._data_embedding = ImageCaptionDataEmbedding()
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.embedding_map = tf.Variable(self._data_embedding.token_embedding_matrix,
                                             dtype=self.model_config.data_type,
                                             trainable=self.model_config.train_embeddings,
                                             name='embedding_map')
        self.input_seq_embeddings = tf.nn.embedding_lookup(params=self.embedding_map,
                                                           ids=self.input_seqs,
                                                           name="input_seq_embeddings")

        pass

    @timeit
    def _build_encoder(self):
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), self.input_images)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_resnet_v2(scaled_input_tensor, is_training=True)
        self.image_features = end_points['PreLogitsFlatten']

        embedding_size = self._data_embedding.embedding_size
        # Map inception output into embedding space.
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=self.image_features,
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
    def __create_rnn_cell(self, num_units):
        rnn_cell = GRUCell(num_units=num_units)
        keep_prob = self.model_config.dropout_keep_prob
        if self.mode == ModeKeys.TRAIN:
            rnn_cell = DropoutWrapper(cell=rnn_cell,
                                      input_keep_prob=keep_prob,
                                      output_keep_prob=keep_prob)
        return rnn_cell
        pass

    @timeit
    @define_scope("decoder")
    def _build_decoder(self):
        num_units = self.model_config.num_hidden_unit
        data_type = self.model_config.data_type
        # put token begin as start state
        token2index = self._data_embedding.token2index
        token_begin = self.model_config.data_config.token_begin
        token_begin_ids = tf.fill([self.batch_size], token2index[token_begin])
        token_begin_embeddings = tf.nn.embedding_lookup(params=self.embedding_map, ids=token_begin_ids)

        # for initial state inputs
        start_embeddings = tf.concat(values=[token_begin_embeddings, self.image_embeddings], axis=-1)
        start_embeddings = tf.nn.l2_normalize(start_embeddings, dim=-1)

        image_embedding_seqs = tf.expand_dims(input=self.image_embeddings, axis=1)
        image_embedding_seqs = tf.tile(input=image_embedding_seqs,
                                       multiples=[1, tf.shape(self.input_seq_embeddings)[1], 1])

        input_embeddings = tf.concat(values=[self.input_seq_embeddings, image_embedding_seqs],
                                     axis=-1)
        input_embeddings = tf.nn.l2_normalize(input_embeddings, dim=-1)
        # input_embeddings = self.input_seq_embeddings

        with tf.variable_scope("rnn_decoder", initializer=self.initializer, reuse=tf.AUTO_REUSE) as rnn_scope:
            # forward RNN cell
            cell_fw = self.__create_rnn_cell(num_units=num_units)
            # backward RNN cell
            cell_bw = self.__create_rnn_cell(num_units=num_units)

            # Feed the image embeddings to set the initial RNN state.
            zero_state_fw = cell_fw.zero_state(batch_size=self.batch_size, dtype=data_type)
            _, initial_state_fw = cell_fw(start_embeddings, zero_state_fw)
            zero_state_bw = cell_bw.zero_state(batch_size=self.batch_size, dtype=data_type)
            _, initial_state_bw = cell_bw(start_embeddings, zero_state_bw)

        if self.mode == ModeKeys.INFER:
            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
            self.initial_states = tf.concat(tf.concat([initial_state_fw, initial_state_bw], axis=-1),
                                            axis=1, name="initial_states")
            # Placeholder for feeding a batch of concatenated states.
            self.state_feeds = tf.placeholder(shape=[None, cell_fw.state_size + cell_bw.state_size],
                                              name="state_feeds", dtype=data_type)

            state_tuple_fw, state_tuple_bw = tf.split(value=self.state_feeds,
                                                      num_or_size_splits=2, axis=1)

            inputs_fw = tf.squeeze(input_embeddings, axis=[1], name="input_fw")
            inputs_bw = tf.squeeze(input_embeddings, axis=[1], name="input_bw")

            # Run a single step for forward and backward.
            outputs_fw, state_tuple_new_fw = cell_fw(inputs=inputs_fw, state=state_tuple_fw)
            outputs_bw, state_tuple_new_bw = cell_bw(inputs=inputs_bw, state=state_tuple_bw)

            outputs = (outputs_fw, outputs_bw)
            final_states = (state_tuple_new_fw, state_tuple_new_bw)

        else:
            # Run the batch of sequence embeddings through the LSTM.
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
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
        self.final_states = tf.concat(final_states, -1)

        pass

    @timeit
    @define_scope("graph")
    def _build_graph(self):
        self._build_encoder()
        self._build_decoder()

    @timeit
    @define_scope(scope_name='optimizer')
    def _build_optimizer(self):
        config = self.model_config
        self.learning_rate = tf.constant(config.learning_rate)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    @timeit
    @define_scope(scope_name='gradients')
    def _build_gradients(self):
        pass

    @timeit
    @define_scope(scope_name='train_op')
    def _build_train_op(self):
        if not self.mode == ModeKeys.INFER:
            self.train_op = self.optimizer.minimize(
                self.loss, name='train_step', global_step=self.global_step_tensor)
