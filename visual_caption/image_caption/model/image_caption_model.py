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
from visual_caption.image_caption.data.data_embedding import ImageCaptionDataEmbedding
from visual_caption.utils.decorator_utils import timeit, define_scope


class ImageCaptionModel(BaseModel):
    def __init__(self, model_config, data_reader, mode):
        super(ImageCaptionModel, self).__init__(
            model_config=model_config,
            data_reader=data_reader,
            mode=mode
        )
        self.data_config = self.data_reader.data_config

    @timeit
    @define_scope(scope_name='inputs')
    def _build_inputs(self):
        data_type = self.model_config.data_type
        dim_visual_feature = self.data_config.dim_visual_feature
        if self.mode == ModeKeys.INFER:
            self.image_feature = tf.placeholder(dtype=data_type,
                                                shape=[None, dim_visual_feature],
                                                name="input_image")
            self.input_feed = tf.placeholder(dtype=tf.int64,
                                             shape=[None],  # batch_size
                                             name="input_feed")
            input_seqs = tf.expand_dims(self.input_feed, 1)
            self.input_seqs = input_seqs
        else:
            (image_id_batch, width_batch, height_batch, depth_batch, image_feature_batch,  # for image
             bbox_shape_batch, bbox_num_batch, bbox_labels, bboxes, bbox_features,  # for bbox
             caption_batch, fw_target_batch, bw_target_batch,  # for text
             caption_ids, fw_target_ids, bw_target_ids,  # for ids
             input_lengths) = self.next_batch

            self.image_ids = image_id_batch
            self.image_feature = image_feature_batch
            self.input_seqs = caption_ids
            self.target_seqs = fw_target_ids
            self.input_lengths = input_lengths

        # input visual features
        # expend_images = tf.expand_dims(self.image_feature, axis=1)
        # self.input_visual_features = tf.concat(name="input_visual_features", axis=1,
        #                                        values=[expend_images, self.region_features])

        # only use image_feature as visual_feature
        self.input_visual_features = self.image_feature

        # replace default model config batch_size with data pipeline batch_size
        self.batch_size = tf.shape(self.image_feature)[0]
        # Maximum decoder time_steps in current batch
        self.max_seq_length = self.model_config.length_max_output

        pass

    @timeit
    @define_scope(scope_name="embeddings")
    def _build_embeddings(self):

        self.token_start_id = self.data_reader.vocabulary.start_id
        self.token_end_id = self.data_reader.vocabulary.end_id
        vocab_num = self.data_reader.vocabulary.num_vocab

        embedding_size = self.data_config.dim_token_feature
        # Save the embedding size in the graph.
        tf.constant(embedding_size, name="embedding_size")

        self.vocab_table = self.data_reader.vocab_table
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.seq_embedding_map = tf.get_variable(
                shape=[vocab_num + 1, embedding_size],
                dtype=self.model_config.data_type,
                initializer=self.emb_initializer,
                trainable=True, name='seq_embedding_map')
            seq_embeddings = tf.nn.embedding_lookup(
                self.seq_embedding_map, self.input_seqs)
        self.input_seq_embeddings = seq_embeddings

        # for token begin batch embeddings
        start_embedding = tf.nn.embedding_lookup(
            self.seq_embedding_map, [self.token_start_id])
        self.start_seq_embeddings = tf.tile(
            input=start_embedding, multiples=[self.batch_size, 1],
            name="start_seq_embeddings")
        # for token end batch embeddings
        end_embedding = tf.nn.embedding_lookup(
            self.seq_embedding_map, [self.token_end_id])
        self.end_seq_embeddings = tf.tile(
            input=end_embedding, multiples=[self.batch_size, 1],
            name="end_seq_embeddings")

        # Mapping visual features into embedding space.
        with tf.variable_scope("visual_embeddings") as visual_embedding_scope:
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

        # # build attention model
        # self.__build_attention()

        # # compute attend visual features
        # self.__attend()

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
    @define_scope(scope_name="decoder")
    def __build_decoder(self):
        """
        decoder for caption outputs
        :return:
        """
        batch_size = self.batch_size
        data_type = self.model_config.data_type
        input_seqs = self.input_seq_embeddings
        input_visuals = self.input_visual_embeddings

        with tf.variable_scope("language",
                               initializer=self.initializer,
                               reuse=tf.AUTO_REUSE) as language_scope:
            language_rnn_cell = self.__create_rnn_cell()
            language_zero_state = language_rnn_cell.zero_state(batch_size, data_type)

            # visual inputs token_start
            language_initial_inputs = tf.nn.l2_normalize(
                tf.concat([input_visuals, self.start_seq_embeddings], axis=-1),
                dim=-1, name="language_initial_inputs")
            _, self.language_initial_state = language_rnn_cell(
                language_initial_inputs, language_zero_state)

            # current input for single step
            seq_visual_embeddings = tf.tile(
                input=tf.expand_dims(input=input_visuals, axis=1),
                multiples=[1, tf.shape(input=input_seqs)[1], 1], name="seq_visual_embeddings")

            language_inputs = tf.nn.l2_normalize(
                tf.concat([seq_visual_embeddings, input_seqs], axis=-1),
                dim=-1, name="language_inputs")

        if self.mode == ModeKeys.INFER:
            # zero state as the initial state for rnn
            self.language_initial_state = language_zero_state
            # current state for single step
            self.language_state_feed = tf.placeholder(
                shape=[None, language_rnn_cell.state_size],
                name="language_state_feed", dtype=data_type)
            # Run a single rnn step for language model.
            language_inputs = tf.squeeze(language_inputs, axis=1)
            language_outputs, self.language_new_state = language_rnn_cell(
                inputs=language_inputs, state=self.language_state_feed)
        else:
            language_outputs, language_states = \
                tf.nn.dynamic_rnn(
                    cell=language_rnn_cell,
                    inputs=language_inputs,
                    sequence_length=self.input_lengths,
                    initial_state=language_zero_state,
                    dtype=data_type, scope=language_scope)
        self.language_outputs = language_outputs

    @timeit
    @define_scope(scope_name="losses")
    def _build_loss(self):
        # Compute logits and weights
        vocab_size = self.data_reader.vocabulary.num_vocab
        outputs = self.language_outputs
        logits = tf.layers.dense(inputs=outputs, units=vocab_size,
                                 kernel_initializer=self.initializer,
                                 name="logits")
        self.predicts = tf.cast(tf.argmax(logits, axis=-1), tf.int32, name="predict")
        if self.mode == ModeKeys.INFER:
            self.softmax = tf.nn.softmax(logits, name="softmax")
            self.predict = tf.cast(tf.argmax(logits, axis=-1), tf.int64, name="predict")
        else:
            weights = tf.sequence_mask(lengths=self.input_lengths,
                                       dtype=outputs.dtype,
                                       name='masks')
            self.mask_weights = weights
            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE) as loss_scope:
                batch_loss = seq2seq.sequence_loss(
                    logits=logits, targets=self.target_seqs, weights=weights)
                self.loss = batch_loss
                tf.losses.add_loss(batch_loss)
                tf.summary.scalar("batch-loss", batch_loss)
            with tf.variable_scope("accuracy", reuse=tf.AUTO_REUSE) as accuracy_scope:
                correct_prediction = tf.equal(self.predicts, self.target_seqs)
                batch_accuracy = tf.div(tf.reduce_sum(
                    tf.multiply(tf.cast(correct_prediction, tf.float32), weights)),
                    tf.reduce_sum(weights), name="batch_accuracy")
                self.accuracy = batch_accuracy
                tf.summary.scalar("accuracy", self.accuracy)

    @timeit
    @define_scope(scope_name='optimizer')
    def _build_optimizer(self):
        config = self.model_config
        # Gradients and SGD update operation for training the model.
        # Arrange for the embedding vars to appear at the beginning.
        # self.learning_rate = tf.constant(config.learning_rate)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            #     self.learning_rate = tf.cond(
            #         self.global_step_tensor < config.start_decay_step,
            #         lambda: self.learning_rate,
            #         lambda: tf.train.exponential_decay(
            #             learning_rate=self.learning_rate,
            #             global_step=(self.global_step_tensor - config.start_decay_step),
            #             decay_steps=config.decay_steps,
            #             decay_rate=config.decay_rate,
            #             staircase=True))
            #     tf.summary.scalar('learning_rate', self.learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            tf.summary.scalar('learning_rate', self.learning_rate)

    @timeit
    @define_scope(scope_name='gradients')
    def _build_gradients(self):
        """Clipping gradients of a model."""
        # if self.mode is not ModeKeys.INFER:
        #     trainables = tf.trainable_variables()
        #     with tf.device(self._get_gpu(self.model_config.num_gpus - 1)):
        #         gradients = tf.gradients(self.loss, trainables)
        #         # clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        #         #     gradients, self.model_config.max_grad_norm)
        #         self._gradients = gradients
        #         # tf.summary.scalar("grad_norm", gradient_norm)
        #         tf.summary.scalar("clipped_gradient", tf.global_norm(gradients))

    @timeit
    @define_scope(scope_name='train_op')
    def _build_train_op(self):
        if self.mode == ModeKeys.TRAIN:
            #     trainables = tf.trainable_variables()
            #     grads_and_vars = zip(self._gradients, trainables)
            #     self.train_op = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars,
            #                                                    global_step=self.global_step_tensor,
            #                                                    name='train_step')
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
