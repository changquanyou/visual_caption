# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import sys
import time

import tensorflow as tf

from visual_caption.base.model.base_model import BaseModel, timeit, define_scope


class ImageCaptionModel(BaseModel):
    def __init__(self, config, data_reader):
        super().__init__(config=config, data_reader=data_reader)

    @timeit
    @define_scope(scope_name='inputs')
    def _build_inputs(self):
        # input images and seqs batch
        batch_size = self._data_reader.data_config.batch_size
        data_config = self._data_reader.data_config

        train_inputs = self._data_reader._build_data_inputs(data_config.train_data_dir)
        test_inputs = self._data_reader._build_data_inputs(data_config.test_data_dir)
        validation_inputs = self._data_reader._build_data_inputs(data_config.validation_data_dir)

        self._embeddings = tf.Variable(self._data_reader.token_embedding_matrix,
                                       dtype=self.config.data_type,
                                       trainable=self.config.train_embeddings,
                                       name='token_embedding')
        self.input_image_embeddings = tf.placeholder(dtype=tf.float32,
                                                     shape=[batch_size, 4096],
                                                     name='image_embeddings')
        if self.config.mode == 'train':
            images_batch, input_seqs_batch, target_seqs_batch, input_mask_batch = train_inputs
        elif self.config.mode == 'test':
            images_batch, input_seqs_batch, target_seqs_batch, input_mask_batch = test_inputs
        elif self.config.mode == 'validation':
            images_batch, input_seqs_batch, target_seqs_batch, input_mask_batch = validation_inputs

        self.input_image_embeddings = images_batch
        self.input_seq_embeddings = tf.nn.embedding_lookup(params=self._embeddings,
                                                           ids=input_seqs_batch,
                                                           name="input_seqs_embeddings")
        self.target_seqs = target_seqs_batch
        self.input_masks = input_mask_batch

    def __create_cell(self):
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

        rnn_cell = self.__create_cell()
        # Feed the image embeddings to set the initial LSTM state.
        zero_state = rnn_cell.zero_state(
            batch_size=self.input_image_embeddings.get_shape()[0], dtype=tf.float32)
        _, initial_state = rnn_cell(self.input_image_embeddings, zero_state)

        """build the Bi_GRU network. Return the y_pred"""
        # cell_fw = tf.contrib.rnn.MultiRNNCell([self.__create_cell() for _ in range(layer_num)], state_is_tuple=True)
        # cell_bw = tf.contrib.rnn.MultiRNNCell([self.__create_cell() for _ in range(layer_num)], state_is_tuple=True)

        sequence_lengths = tf.reduce_sum(self.input_masks, 1)

        cell_fw = self.__create_cell()
        cell_bw = self.__create_cell()
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
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

    def _build_fetches(self):
        self.fetches = [self._train_op, self._cost, self._merged]
        return self.fetches

    def run_train(self):
        print("......begin training......")
        with tf.Session(config=self.config.sess_config) as sess:
            self._summary_writer.add_graph(sess.graph)
            epoch_size = self.config.max_max_epoch
            checkpoint_dir = self.config.checkpoint_dir
            saver = tf.train.Saver()

            # CheckPoint State
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
            if checkpoint:
                self.logger.info("Loading model parameters from {}".format(checkpoint.model_checkpoint_path))
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            else:
                self.logger.info("Created model with fresh parameters.")
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            train_fetches = self._build_fetches()
            try:
                start = time.time()
                # batch_count = 0
                while not coord.should_stop():
                    global_step = tf.train.global_step(sess, self._global_step)
                    _, batch_loss, batch_summary = sess.run(fetches=train_fetches)
                    self._summary_writer.add_summary(batch_summary, global_step=global_step)
                    # batch_count += 1

                    if global_step % 100 == 0 and global_step > 0:
                        last = time.time() - start
                        print('global_step={}, loss={}, time={}'.format(global_step, batch_loss, last))
                    if global_step % 1000 == 0 and global_step > 0:
                        self._save_model(sess=sess, global_step=global_step)
                        last = time.time() - start
                        print('global_step={}, loss={}, time={}'.format(global_step, batch_loss, last))

            except tf.errors.OutOfRangeError:
                print("Done training after reading all data")
            except Exception as exception:
                print(exception)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
            finally:
                # finalise
                coord.request_stop()  # Stop the threads
                coord.join(threads)  # Wait for threads to stop

            self._summary_writer.close()
        print("......end training.....")

    @timeit
    @define_scope(scope_name='losses')
    def _build_loss(self):
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
        pass
