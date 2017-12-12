# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib import rnn
from tensorflow.contrib.learn import ModeKeys

from visual_caption.base.model.base_model import BaseModel
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
        dim_visual_feature = self.model_config.data_config.dim_visual_feature

        # A float32 Tensor with shape [1]
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')

        if self.mode == ModeKeys.INFER:
            # In inference mode, images and inputs are fed via placeholders.
            self.image_feature = tf.placeholder(dtype=data_type, name='image_feature',
                                                shape=[None, dim_visual_feature])
            self.region_features = tf.placeholder(dtype=data_type, name='region_features',
                                                  shape=[None, None, dim_visual_feature])

            self.input_feed = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed")
            input_seqs = tf.expand_dims(self.input_feed, 1)
            self.input_seqs = input_seqs
        else:
            id_batch, width_batch, height_batch, depth_batch, feature_batch, \
            bbox_shape_batch, bbox_num, bbox_labels, bboxes, bbox_features, \
            caption_batch, target_batch, caption_ids, target_ids, caption_lengths, target_lengths = self.next_batch

            self.image_ids = id_batch

            self.image_feature = feature_batch
            self.region_features = bbox_features

            self.input_seqs = caption_ids
            self.target_seqs = target_ids
            self.input_lengths = caption_lengths
            self.target_lengths = target_lengths

            # input visual features
        expend_images = tf.expand_dims(self.image_feature, axis=1)

        self.input_visual_features = tf.concat(name="input_visual_features", axis=1,
                                               values=[expend_images, self.region_features])
        # replace default model_config batch_size with data pipeline batch_size
        self.batch_size = tf.shape(self.image_feature)[0]
        # self.num_regions = tf.shape(self.input_visual_features)[0]
        # Maximum decoder time_steps in current batch
        if self.mode == ModeKeys.INFER:
            self.num_caption_max_length = self.model_config.data_config.num_caption_max_length
        else:
            self.num_caption_max_length = tf.reduce_max(self.input_lengths, axis=-1)

        pass

    @timeit
    @define_scope(scope_name="embeddings")
    def _build_embeddings(self):

        self.data_config = self.model_config.data_config
        self.token_start_id = self.data_reader.vocabulary.start_id
        self.token_end_id = self.data_reader.vocabulary.end_id
        vocab_num = self.data_reader.vocabulary.num_vocab

        embedding_size = self.model_config.data_config.dim_token_feature
        # Save the embedding size in the graph.
        tf.constant(embedding_size, name="embedding_size")

        self.vocab_table = self.data_reader.vocab_table
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.seq_embedding_map = tf.get_variable(
                shape=[vocab_num + 1, embedding_size],
                dtype=self.model_config.data_type,
                initializer=self.emb_initializer,
                trainable=True,
                name='seq_embedding_map')
            seq_embeddings = tf.nn.embedding_lookup(
                self.seq_embedding_map, self.input_seqs)

            # current_token_embedding = tf.nn.embedding_lookup(self.seq_embedding_map, self.current_token)
            # self.current_token_embedding = current_token_embedding

        self.input_seq_embeddings = seq_embeddings

        # for token begin batch embeddings
        start_embedding = tf.nn.embedding_lookup(
            self.seq_embedding_map, [self.token_start_id])
        self.start_seq_embeddings = tf.tile(input=start_embedding,
                                            multiples=[self.batch_size, 1],
                                            name="start_seq_embeddings")
        # for token end batch embeddings
        end_embedding = tf.nn.embedding_lookup(
            self.seq_embedding_map, [self.token_end_id])
        self.end_seq_embeddings = tf.tile(input=end_embedding,
                                          multiples=[self.batch_size, 1],
                                          name="end_seq_embeddings")

        # Mapping visual features into embedding space.
        with tf.variable_scope("visual_embeddings") as scope:
            visual_embeddings = tf.layers.dense(
                inputs=self.input_visual_features,
                units=embedding_size)
        self.input_visual_embeddings = tf.nn.l2_normalize(
            visual_embeddings, dim=-1, name="input_visual_embeddings")
        pass

    @timeit
    @define_scope(scope_name="graph")
    def _build_graph(self):
        self.__build_encoder()

        # build attention model
        self.__build_attention()

        # compute attend visual features
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
        num_units = self.model_config.num_hidden_unit
        rnn_cell = rnn.GRUCell(num_units=num_units)
        if self.mode == ModeKeys.TRAIN:
            dropout_keep = self.model_config.dropout_keep_prob
            rnn_cell = rnn.DropoutWrapper(
                cell=rnn_cell,
                input_keep_prob=dropout_keep,
                output_keep_prob=dropout_keep)
        return rnn_cell

    @timeit
    def __build_attention(self):
        """
        build attention model
        :return:
        """
        batch_size = self.batch_size
        data_type = self.model_config.data_type

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

            # expend and tile visual_feature according with input seq
            seq_length = tf.shape(self.input_seq_embeddings)[1]
            seq_average_visuals = tf.tile(tf.expand_dims(average_visuals, axis=1),
                                          multiples=[1, seq_length, 1])

            # attention input is the fusion of visual and text features
            # for inference, this input is the fusion of visual and partial caption
            # for train and valid, the input is the fusion of visual and whole fusion
            # shape [batch, seq_length, dim_visual_feature]
            attend_inputs = tf.concat([seq_average_visuals, self.input_seq_embeddings], axis=-1)
            attend_inputs = tf.nn.l2_normalize(attend_inputs, name="attend_inputs", dim=-1)

            if self.mode == ModeKeys.INFER:
                # for infer model, batch size is 1
                # Placeholder for feeding a batch of concatenated states.
                self.attend_fw_initial_state = attend_fw_initial_state
                self.attend_bw_initial_state = attend_bw_initial_state

                # Placeholder for feeding a batch of concatenated states.
                self.attend_fw_state_feed = tf.placeholder(shape=[None, attend_fw_rnn_cell.state_size],
                                                           name="attend_fw_state_feed", dtype=data_type)
                self.attend_bw_state_feed = tf.placeholder(shape=[None, attend_bw_rnn_cell.state_size],
                                                           name="attend_bw_state_feed", dtype=data_type)

                attend_input_fw = tf.squeeze(attend_inputs, axis=[1], name="attend_input_fw")
                attend_input_bw = tf.squeeze(attend_inputs, axis=[1], name="attend_input_bw")

                # Run one single attention step forward
                attention_fw_outputs, attention_fw_new_state = attend_fw_rnn_cell(
                    inputs=attend_input_fw, state=self.attend_fw_state_feed)

                # Run one single attention step backward
                attention_bw_outputs, attention_bw_new_state = attend_bw_rnn_cell(
                    inputs=attend_input_bw, state=self.attend_bw_state_feed)

                self.attention_fw_outputs = tf.expand_dims(input=attention_fw_outputs, dim=0)
                self.attention_bw_outputs = tf.expand_dims(input=attention_bw_outputs, dim=0)

                attend_outputs = tf.add_n([self.attention_fw_outputs, self.attention_bw_outputs],
                                          name="attend_outputs")


            else:
                attend_outputs, attend_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=attend_fw_rnn_cell,
                    cell_bw=attend_bw_rnn_cell,
                    inputs=attend_inputs,
                    sequence_length=self.input_lengths,
                    initial_state_fw=self.attend_fw_initial_state,
                    initial_state_bw=self.attend_bw_initial_state,
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
         compute the attend weighted visual features based on attend_outputs
        """
        data_type = self.model_config.data_type
        num_visual_features = self.data_config.num_visual_features
        dim_hidden = 1000

        # expand to sequence length,
        # shape = [batch, seq_length, num_regions, dim_visual_embedding]
        seq_visual_embeddings = tf.tile(
            tf.expand_dims(self.input_visual_embeddings, axis=1),
            multiples=[1, self.num_caption_max_length, 1, 1],
            name="seq_visual_embeddings")
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
        fused_attends = tf.reshape(fused_attends, shape=[-1, dim_hidden])

        # mapping backward from hidden space
        w_a = tf.Variable(tf.random_normal([1, dim_hidden]), dtype=data_type)
        attends = tf.matmul(w_a, fused_attends, transpose_b=True)
        attends = tf.reshape(attends, shape=(batch_size, seq_length, num_visual_features))

        # shape [batch, seq_max_length, num_visual]
        attend_weights = tf.nn.softmax(attends)
        attend_weights = tf.tile(tf.expand_dims(attend_weights, axis=-1),
                                 multiples=[1, 1, 1, tf.shape(seq_visual_embeddings)[-1]])

        # shape = [batch, seq_length, dim_visual_embedding]
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
            _, self.language_fw_initial_state = language_rnn_fw_cell(
                language_fw_initial_inputs, language_fw_zero_state)

            # language rnn_cell backward initial
            # shape = [batch, dim_visual_embedding + dim_attention]
            language_bw_initial_inputs = tf.nn.l2_normalize(
                tf.concat([self.attend_bw_initial_visuals, self.attend_bw_initial_outputs], axis=-1),
                dim=-1, name="language_bw_initial_inputs")
            language_bw_zero_state = language_rnn_bw_cell.zero_state(batch_size, data_type)
            _, self.language_bw_initial_state = language_rnn_bw_cell(
                language_bw_initial_inputs, language_bw_zero_state)

        if self.mode == ModeKeys.INFER:
            # In inference mode, states for feeding and fetching.
            language_fw_state_feed = tf.placeholder(shape=[None, language_rnn_fw_cell.state_size],
                                                    name="language_fw_state_feed", dtype=data_type)
            language_bw_state_feed = tf.placeholder(shape=[None, language_rnn_bw_cell.state_size],
                                                    name="language_bw_state_feed", dtype=data_type)

            attend_visuals = tf.squeeze(self.attend_visuals, axis=0)
            attention_fw_outputs = tf.squeeze(self.attention_fw_outputs, axis=0)
            attention_bw_outputs = tf.squeeze(self.attention_bw_outputs, axis=0)

            language_input_fw = tf.nn.l2_normalize(
                tf.concat([attend_visuals, attention_fw_outputs], axis=-1),
                dim=-1, name="language_fw_input")

            language_input_bw = tf.nn.l2_normalize(
                tf.concat([attend_visuals, attention_bw_outputs], axis=-1),
                dim=-1, name="language_bw_input")

            # Run one single language step.
            language_fw_outputs, language_fw_new_state = language_rnn_fw_cell(
                inputs=language_input_fw, state=language_fw_state_feed)

            language_bw_outputs, language_bw_new_state = language_rnn_bw_cell(
                inputs=language_input_bw, state=language_bw_state_feed)

            language_outputs = (language_fw_outputs, language_bw_outputs)
            language_new_states = (language_fw_new_state, language_bw_new_state)

        else:
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
                initial_state_fw=self.language_fw_initial_state,
                initial_state_bw=self.language_bw_initial_state,
                dtype=data_type)

        self.decoder_outputs = tf.concat(
            values=language_outputs, axis=-1, name="language_outputs")

    @timeit
    @define_scope(scope_name="losses")
    def _build_loss(self):
        # Compute logits and weights
        vocab_size = self.data_reader.vocabulary.num_vocab
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
            batch_loss = seq2seq.sequence_loss(logits=logits,
                                               targets=self.target_seqs,
                                               weights=weights,
                                               name="sequence_loss")
            self.loss = batch_loss
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()
            # Add summaries.
            tf.summary.scalar("batch-loss", batch_loss)
            tf.summary.scalar("total-loss", total_loss)
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
        """
        decode:
            p(y(t)|y(1:t-1)) = softmax( W(p)* h_language(t)+b(p) )
        :return:
        """
        assert self.mode == ModeKeys.INFER

    pass

    pass
