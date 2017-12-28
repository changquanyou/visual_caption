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
            model_config=model_config,
            data_reader=data_reader,
            mode=mode
        )

    @timeit
    @define_scope(scope_name='inputs')
    def _build_inputs(self):
        data_type = self.model_config.data_type
        if self.mode == ModeKeys.INFER:
            # self.image_ids = tf.placeholder(dtype=tf.string, shape=[None], name='image_ids')
            self.input_image_embeddings = tf.placeholder(dtype=data_type,
                                                         shape=[None, 4096],
                                                         name="image_inputs_embeddings")
            # In inference mode, images and inputs are fed via placeholders.
            # image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            self.input_feed = tf.placeholder(dtype=tf.int64,
                                             shape=[None],  # batch_size
                                             name="input_feed")
            input_seqs = tf.expand_dims(self.input_feed, 1)
            # A float32 Tensor with shape [batch_size, image_feature_size].
            self.image_feature = tf.placeholder(tf.float32, [None, 4096], name='image_feature')
            self.input_seqs = input_seqs

            # self.input_seqs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_seqs')
            # self.target_seqs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='target_seqs')
            #
            # self.input_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='input_lengths')
            # self.target_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='target_lengths')
        else:

            (image_ids, image_features, captions, targets,
             caption_ids, target_ids, caption_lengths, target_lengths) = self.next_batch

            self.input_image_embeddings = image_features
            self.image_ids = image_ids

            self.input_seqs = caption_ids
            self.target_seqs = target_ids

            self.input_lengths = caption_lengths
            self.target_lengths = target_lengths

        # replace default model config batch_size with data pipeline batch_size
        self.batch_size = tf.shape(self.input_image_embeddings)[0]
        # Maximum decoder time_steps in current batch
        self.max_seq_length = self.model_config.length_max_output

        pass

    @timeit
    @define_scope(scope_name='embeddings')
    def _build_embeddings(self):
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.embedding_map = tf.Variable(self._data_embedding.token_embedding_matrix,
                                             dtype=self.model_config.data_type,
                                             trainable=self.model_config.train_embeddings,
                                             name='embedding_map')
        self.input_seq_embeddings = tf.nn.embedding_lookup(
            params=self.embedding_map, ids=self.input_seqs)
        self.target_seq_embeddings = tf.nn.embedding_lookup(
            params=self.embedding_map, ids=self.target_seqs)
        pass

    @timeit
    @define_scope(scope_name='graph')
    def _build_graph(self):
        self._build_encoder()
        self._build_decoder()

    @timeit
    def __create_rnn_cell(self, num_units):
        rnn_cell = GRUCell(num_units=num_units)
        if self.mode == "train":
            rnn_cell = DropoutWrapper(
                rnn_cell,
                input_keep_prob=self.model_config.dropout_keep_prob,
                output_keep_prob=self.model_config.dropout_keep_prob)
        return rnn_cell

    @timeit
    @define_scope(scope_name='encoder')
    def _build_encoder(self):

        data_type = self.model_config.data_type
        num_hidden_unit = self.model_config.num_hidden_unit

        image_embedding_seqs = tf.expand_dims(input=self.input_image_embeddings, axis=1)
        image_embedding_seqs = tf.tile(image_embedding_seqs,
                                       multiples=[1, self.max_seq_length, 1])
        # image_seq_embeddings = tf.concat(values=[self.input_seq_embeddings, image_embedding_seqs],
        #                                  axis=-1, name="image_seq_embeddings")
        # encoder_inputs = self.input_seq_embeddings
        encoder_inputs = image_embedding_seqs

        # Feed the image embeddings to set the initial RNN state.
        rnn_cell = self.__create_rnn_cell(num_hidden_unit)
        zero_state = rnn_cell.zero_state(batch_size=self.batch_size,
                                         dtype=tf.float32)
        _, initial_state = rnn_cell(self.input_image_embeddings, zero_state)
        # forward RNN cell
        cell_fw = self.__create_rnn_cell(num_hidden_unit)
        # backward RNN cell
        cell_bw = self.__create_rnn_cell(num_hidden_unit)
        outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=encoder_inputs,
            # sequence_length=self.input_lengths,
            initial_state_fw=initial_state,
            initial_state_bw=initial_state,
            dtype=data_type
        )

        self.encoder_outputs = tf.concat(outputs, -1)
        self.encoder_final_state = tf.concat(outputs_states, -1)

    @timeit
    @define_scope(scope_name='decoder')
    def _build_decoder(self):

        encoder_outputs = self.encoder_outputs
        encoder_final_state = self.encoder_final_state

        batch_size = self.batch_size
        num_units = self.model_config.num_hidden_unit
        beam_width = self.model_config.beam_width
        vocab_size = self._data_embedding.vocab_size

        token2index = self._data_embedding.token2index
        token_begin = self.model_config.data_config.token_begin
        token_end = self.model_config.data_config.token_end

        start_tokens = [token2index[token_begin]]
        end_token = token2index[token_end]
        decoder_embedding = self.embedding_map

        decoder_cell = self.__create_rnn_cell(num_units * 2)
        decoder_initial_state = encoder_final_state
        # decoder_cell, decoder_initial_state = self._attention_cell(
        #     decoder_cell=decoder_cell,
        #     encoder_outputs=encoder_outputs,
        #     decoder_initial_state=decoder_initial_state,
        #     batch_size=batch_size)
        # Output projection layer to convert cell_outputs to logits
        self.output_layer = Dense(vocab_size, name='output_projection')
        if self.mode == ModeKeys.INFER:
            helper = GreedyEmbeddingHelper(embedding=decoder_embedding,
                                           start_tokens=tf.fill([batch_size],
                                                                token2index[token_begin]),
                                           end_token=end_token)
        else:  # for train or eval helper
            target_lengths = self.target_lengths
            # image_embedding_seqs = tf.expand_dims(input=self.input_image_embeddings, axis=1)
            # image_embedding_seqs = tf.tile(image_embedding_seqs,
            #                                multiples=[1, tf.reduce_max(target_lengths), 1])
            # target_seq_embeddings = tf.concat(values=[self.target_seq_embeddings,
            #                                           image_embedding_seqs],
            #                                   axis=-1)

            target_seq_embeddings = self.target_seq_embeddings
            # target_seq_embeddings = tf.nn.l2_normalize(target_seq_embeddings, dim=-1)

            helper = seq2seq.TrainingHelper(inputs=target_seq_embeddings,
                                            sequence_length=target_lengths,
                                            name='training_helper')

        decoder = seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                       initial_state=decoder_initial_state,
                                       output_layer=self.output_layer)

        outputs, output_states, output_lengths = seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=self.max_seq_length)

        if beam_width > 0:
            logits = tf.no_op()
            sample_id = outputs.predicted_ids
        else:
            logits = outputs.rnn_output
            sample_id = outputs.sample_id

        self.decoder_outputs = logits
        self.decoder_predictions = sample_id
        pass

    @timeit
    @define_scope(scope_name="losses")
    def _build_loss(self):
        # Compute logits and weights
        # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
        with tf.variable_scope('output'):
            self.logits = self.decoder_outputs
            if not self.mode == ModeKeys.INFER:
                weights = tf.sequence_mask(lengths=self.target_lengths,
                                           maxlen=tf.reduce_max(self.target_lengths),
                                           dtype=self.decoder_outputs.dtype,
                                           name='masks')
                batch_loss = seq2seq.sequence_loss(logits=self.logits,
                                                   targets=self.target_seqs,
                                                   weights=weights)
                tf.losses.add_loss(batch_loss)
                total_loss = tf.losses.get_total_loss()
                self.loss = batch_loss
                self.total_loss = total_loss

                # Add summaries.
                tf.summary.scalar("batch_loss", batch_loss)
                tf.summary.scalar("total_loss", total_loss)
                correct_prediction = tf.equal(self.decoder_predictions, self.target_seqs)

                batch_accuracy = tf.div(tf.reduce_sum(
                    tf.multiply(tf.cast(correct_prediction, tf.float32), weights)),
                    tf.reduce_sum(weights), name="batch_accuracy")
                self.accuracy = batch_accuracy
                tf.summary.scalar("accuracy", self.accuracy)


class ImageCaptionFullModel(ImageCaptionModel):
    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionFullModel, self).__init__(
            model_config, data_reader, mode)

    @timeit
    @define_scope('inputs')
    def _build_inputs(self):
        data_type = self.model_config.data_type
        visual_feature_size = 1536
        # A float32 Tensor with shape [1]
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')

        if self.mode == ModeKeys.INFER:

            # In inference mode, images and inputs are fed via placeholders.
            self.image_feature = tf.placeholder(tf.float32, [None, visual_feature_size], name='image_feature')
            self.input_feed = tf.placeholder(tf.int64, [None], name="input_feed")
            input_seqs = tf.expand_dims(self.input_feed, 1)
            # A float32 Tensor with shape [batch_size, image_feature_size].

            self.input_seqs = input_seqs
        else:
            (image_ids, image_features, captions, targets,
             caption_ids, target_ids, caption_lengths, target_lengths) = self.next_batch

            self.image_ids = image_ids
            self.image_feature = image_features

            self.input_seqs = caption_ids
            self.input_lengths = caption_lengths

            self.target_seqs = target_ids
            self.target_lengths = target_lengths

        self.batch_size = tf.shape(self.image_feature)[0]

    @timeit
    @define_scope("embeddings")
    def _build_embeddings(self):
        self._data_embedding = ImageCaptionDataEmbedding()
        self.embedding_size = self._data_embedding.embedding_size
        self.vocab_size = self._data_embedding.vocab_size
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.embedding_map = tf.Variable(self._data_embedding.token_embedding_matrix,
                                             dtype=self.model_config.data_type,
                                             trainable=self.model_config.train_embeddings,
                                             name='embedding_map')
        input_seq_embeddings = tf.nn.embedding_lookup(self.embedding_map,
                                                      self.input_seqs)
        self.input_seq_embeddings = input_seq_embeddings

        # Map inception output into embedding space.
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=self.image_feature,
                num_outputs=self.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

        # Save the embedding size in the graph.
        tf.constant(self.embedding_size, name="embedding_size")
        self.image_embeddings = image_embeddings

    def _build_graph(self):
        self.__build_encoder()
        self.__build_decoder()
        pass

    @define_scope(scope_name="encoder")
    def __build_encoder(self):
        # just use the extracted inception_resnet features from images
        pass

    @timeit
    @define_scope(scope_name="decoder")
    def __build_decoder(self):
        num_units = self.model_config.num_hidden_unit
        keep_prob = self.model_config.dropout_keep_prob
        rnn_cell = rnn.BasicLSTMCell(num_units=num_units,
                                     state_is_tuple=True)
        if self.mode == ModeKeys.TRAIN:
            rnn_cell = rnn.DropoutWrapper(cell=rnn_cell,
                                          input_keep_prob=keep_prob,
                                          output_keep_prob=keep_prob)

        with tf.variable_scope("RNN", initializer=self.initializer) as rnn_scope:
            # Feed the image embeddings to set the initial LSTM state.
            zero_state = rnn_cell.zero_state(batch_size=self.batch_size,
                                             dtype=tf.float32)
            _, initial_state = rnn_cell(self.image_embeddings, zero_state)

        # Allow the RNN variables to be reused.
        rnn_scope.reuse_variables()

        if self.mode == ModeKeys.INFER:
            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
            self.initial_state = tf.concat(values=initial_state,
                                           axis=1,
                                           name="initial_state")
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
            outputs, final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=self.input_seq_embeddings,
                sequence_length=self.input_lengths,
                initial_state=initial_state,
                dtype=tf.float32,
                scope=rnn_scope)
        # Stack batches vertically.
        # self.outputs = tf.reshape(outputs, [-1, rnn_cell.output_size])
        self.outputs = outputs
        pass

    @timeit
    @define_scope(scope_name="losses")
    def _build_loss(self):
        # Compute logits and weights
        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=self.outputs,
                num_outputs=self.vocab_size,
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
