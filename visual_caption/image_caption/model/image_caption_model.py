# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import GRUCell, DropoutWrapper

from visual_caption.base.model.base_model import BaseModel
from visual_caption.image_caption.data.data_embedding import ImageCaptionDataEmbedding
from visual_caption.utils.decorator_utils import timeit, define_scope


class ImageCaptionModel(BaseModel):
    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionModel, self).__init__(
            model_config=model_config,
            data_reader=data_reader,
            mode=mode
        )

    @timeit
    @define_scope(scope_name='embeddings')
    def _build_embeddings(self):
        self._data_embedding = ImageCaptionDataEmbedding()

        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.embedding_map = tf.Variable(self._data_embedding.token_embedding_matrix,
                                             dtype=self.model_config.data_type,
                                             trainable=self.model_config.train_embeddings,
                                             name='embedding_map')
            if not self.mode == ModeKeys.INFER:
                self.input_seq_embeddings = tf.nn.embedding_lookup(
                    params=self.embedding_map, ids=self.input_seqs)
                self.target_seq_embeddings = tf.nn.embedding_lookup(
                    params=self.embedding_map, ids=self.target_seqs)

        pass

    @timeit
    @define_scope(scope_name='inputs')
    def _build_inputs(self):
        data_type = self.model_config.data_type
        if self.mode == ModeKeys.INFER:
            # self.image_ids = tf.placeholder(dtype=tf.string, shape=[None], name='image_ids')
            self.input_image_embeddings = tf.placeholder(dtype=data_type,
                                                         shape=[None, 4096],
                                                         name="image_inputs_embeddings")

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

        with tf.variable_scope("encoder_cell", initializer=self.initializer) as encoder_rnn:
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

        # Allow the encoder_rnn variables to be reused.
        encoder_rnn.reuse_variables()

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

        # Output projection layer to convert cell_outputs to logits
        with tf.variable_scope("decoder/output_projection"):
            self.output_layer = Dense(vocab_size, name='output_projection')

        with tf.variable_scope("decoder_cell", initializer=self.initializer) as decoder_rnn_scope:
            decoder_cell = self.__create_rnn_cell(num_units * 2)
            decoder_initial_state = encoder_final_state
            # if attention mechanism is given, wrap cell with attention mechanism
            # if self.model_config.attention_mechanism:
            #     decoder_cell, decoder_initial_state = self._attention_cell(
            #         decoder_cell=decoder_cell,
            #         encoder_outputs=encoder_outputs,
            #         decoder_initial_state=decoder_initial_state,
            #         batch_size=batch_size
            #     )
                # Allow the encoder_rnn variables to be reused.

        decoder_rnn_scope.reuse_variables()

        image_embedding_seqs = tf.expand_dims(input=self.input_image_embeddings, axis=1)
        image_embedding_seqs = tf.tile(image_embedding_seqs,
                                       multiples=[1, self.max_seq_length, 1])
        image_embedding_seqs = tf.multiply(image_embedding_seqs, 0.001)

        # target_seq_embeddings = tf.concat(values=[self.target_seq_embeddings, image_embedding_seqs],
        #                                   axis=-1, name="target_seq_embeddings")


        with tf.variable_scope('decoder_helper', reuse=True):
            if self.mode == ModeKeys.INFER:  # for inference
                helper = GreedyEmbeddingHelper(embedding=decoder_embedding,
                                               start_tokens=tf.fill([batch_size],
                                                                    token2index[token_begin]),
                                               end_token=end_token)
            else:  # for train or eval helper
                target_lengths = self.target_lengths
                target_seq_embeddings = self.target_seq_embeddings
                helper = seq2seq.TrainingHelper(inputs=target_seq_embeddings,
                                                sequence_length=target_lengths,
                                                name='training_helper')

            if self.mode == ModeKeys.INFER and beam_width > 0:
                decoder = seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                    embedding=decoder_embedding,
                                                    start_tokens=tf.fill([batch_size], token2index[token_begin]),
                                                    end_token=end_token,
                                                    beam_width=beam_width,
                                                    initial_state=decoder_initial_state,
                                                    output_layer=self.output_layer,
                                                    length_penalty_weight=0.0)
            else:
                decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                               helper=helper,
                                               initial_state=decoder_initial_state,
                                               output_layer=self.output_layer)

        outputs, output_states, output_lengths = seq2seq.dynamic_decode(
            decoder=decoder, maximum_iterations=self.max_seq_length,
            scope="dynamic_decode"
        )

        if beam_width > 0:
            logits = tf.no_op()
            sample_id = outputs.predicted_ids
        else:
            logits = outputs.rnn_output
            sample_id = outputs.sample_id

        self.decoder_outputs = logits
        self.decoder_predictions = sample_id
        pass

    def _attention_cell(self,
                        decoder_cell,
                        encoder_outputs,
                        decoder_initial_state,
                        batch_size):

        source_sequence_length = self.input_lengths

        dtype = self.encoder_outputs.dtype

        attention_mechanism = create_attention_mechanism(
            attention_option=self.model_config.attention_mechanism,
            num_units=self.model_config.num_attention_unit,
            memory=encoder_outputs,
            source_sequence_length=source_sequence_length,
            mode=self.mode
        )

        attention_cell = seq2seq.AttentionWrapper(
            cell=decoder_cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.model_config.num_attention_layer,
            name="decoder_attention_cell"
        )
        _state = attention_cell.zero_state(dtype=dtype,
                                           batch_size=batch_size)
        attention_cell_initial_state = _state.clone(cell_state=decoder_initial_state)

        return attention_cell, attention_cell_initial_state

    def _beamsearch_attention_cell(self,
                                   encoder_outputs,
                                   decoder_cell,
                                   decoder_initial_state,
                                   beam_width,
                                   batch_size):
        true_batch_size = batch_size
        dtype = self.encoder_outputs.dtype

        tiled_sequence_length = seq2seq.tile_batch(
            self.input_lengths, multiplier=beam_width)
        tiled_encoder_outputs = seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)

        if self.model_config.attention_mechanism:
            attention_mechanism = create_attention_mechanism(
                attention_option=self.model_config.attention_mechanism,
                num_units=self.model_config.num_attention_unit,
                memory=tiled_encoder_outputs,
                source_sequence_length=tiled_sequence_length,
                mode=self.mode
            )
            #
            decoder_cell = seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.model_config.num_attention_layer,
                name="decoder_attention_cell"
            )

        _state = decoder_cell.zero_state(
            dtype=dtype, batch_size=true_batch_size * beam_width)

        tiled_decoder_initial_state = seq2seq.tile_batch(
            decoder_initial_state, multiplier=beam_width)
        decoder_initial_state = tiled_decoder_initial_state

        return decoder_cell, decoder_initial_state
        pass

    @timeit
    @define_scope(scope_name="losses")
    def _build_loss(self):
        # Compute logits and weights

        # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]


        # _output_layer = Dense(vocab_size, name='output_layer')
        with tf.variable_scope('output'):
            self.logits = self.decoder_outputs

            if not self.mode==ModeKeys.INFER:
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


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length, mode):
    """Create attention mechanism based on the attention_option."""
    del mode  # unused

    # Mechanism
    if attention_option == "luong":
        attention_mechanism = seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism
