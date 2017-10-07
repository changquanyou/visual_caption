# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import numpy as np

import tensorflow as tf

from visual_caption.base.data.base_data_reader import BaseDataReader
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from gensim.models.word2vec import Word2Vec

# Data Reader class for AI_Challenge_2017
class ImageCaptionDataReader(BaseDataReader):
    """
        Read data for train, validation, test dataset with embedding model
    """

    def __init__(self, data_config=ImageCaptionDataConfig()):
        super().__init__(data_config=data_config)

        self.tf_reader = tf.TFRecordReader()

        self.context_features = {
            self.data_config.visual_image_id_name: tf.FixedLenFeature([], dtype=tf.string),
            self.data_config.visual_feature_name: tf.FixedLenFeature([], dtype=tf.string),
        }

        self.sequence_features = {
            self.data_config.caption_text_name: tf.FixedLenSequenceFeature([], dtype=tf.string),
            self.data_config.caption_ids_name: tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        self.load_embeddings()

    def load_embeddings(self):
        """
        load char2vec or word2vec model for token embeddings
        :return:
        """
        self.token2vec = Word2Vec.load(self.data_config.char2vec_model)
        self.vocab = self.token2vec.wv.vocab

        self.token2index = {}
        self.index2token = {}
        self.token_embedding_matrix = np.zeros([len(self.vocab) + 1, self.data_config.embedding_dim_size])
        for idx, token in enumerate(self.token2vec.wv.index2word):
            token_embedding = self.token2vec.wv[token]
            self.index2token[idx] = token
            self.token2index[token] = idx
            self.token_embedding_matrix[idx] = token_embedding

        # for unknown token
        self.token2index[self.data_config.unknown_token] = len(self.vocab)
        self.vocab_size = len(self.token2index)

        pass

    pass

    def _batch_with_dynamic_pad(self, images_and_captions,
                                batch_size,
                                queue_capacity,
                                add_summaries=True):
        """Batches input images and captions.

        This function splits the caption into an input sequence and a target sequence,
        where the target sequence is the input sequence right-shifted by 1. Input and
        target sequences are batched and padded up to the maximum length of sequences
        in the batch. A mask is created to distinguish real words from padding words.

        Example:
          Actual captions in the batch ('-' denotes padded character):
            [
              [ 1 2 3 4 5 ],
              [ 1 2 3 4 - ],
              [ 1 2 3 - - ],
            ]

          input_seqs:
            [
              [ 1 2 3 4 ],
              [ 1 2 3 - ],
              [ 1 2 - - ],
            ]

          target_seqs:
            [
              [ 2 3 4 5 ],
              [ 2 3 4 - ],
              [ 2 3 - - ],
            ]

          mask:
            [
              [ 1 1 1 1 ],
              [ 1 1 1 0 ],
              [ 1 1 0 0 ],
            ]

        Args:
          images_and_captions: A list of pairs [image, caption], where image is a
            Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
            any length. Each pair will be processed and added to the queue in a
            separate thread.
          batch_size: Batch size.
          queue_capacity: Queue capacity.
          add_summaries: If true, add caption length summaries.

        Returns:
          images: A Tensor of shape [batch_size, height, width, channels].
          input_seqs: An int32 Tensor of shape [batch_size, padded_length].
          target_seqs: An int32 Tensor of shape [batch_size, padded_length].
          mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
        """
        enqueue_list = []
        for image, caption in images_and_captions:
            caption_length = tf.shape(caption)[0]
            input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

            input_seq = tf.slice(caption, [0], input_length)
            target_seq = tf.slice(caption, [1], input_length)
            indicator = tf.ones(input_length, dtype=tf.int32)
            enqueue_list.append([image, input_seq, target_seq, indicator])

        images, input_seqs, target_seqs, mask = tf.train.batch_join(
            enqueue_list,
            batch_size=batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch_and_pad")

        if add_summaries:
            lengths = tf.add(tf.reduce_sum(mask, 1), 1)
            tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
            tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
            tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

        return images, input_seqs, target_seqs, mask

    def _parse_sequence_example(self, serialized_example):
        # parsing sequence example
        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features=self.context_features,
            sequence_features=self.sequence_features
        )

        image_id = context[self.data_config.visual_image_id_name]
        visual_feature = context[self.data_config.visual_feature_name]

        # for vgg19 fc7
        visual_feature = tf.decode_raw(visual_feature, tf.float32)
        image_feature = tf.reshape(visual_feature, [4096])

        caption_text = sequence[self.data_config.caption_text_name]
        caption_ids = sequence[self.data_config.caption_ids_name]

        return image_feature, caption_ids

    def _prefetch_input_data(self, reader,
                             data_dir,
                             is_training,
                             batch_size,
                             values_per_shard,
                             input_queue_capacity_factor=16,
                             num_reader_threads=1,
                             shard_queue_name="filename_queue",
                             value_queue_name="input_queue"):
        """Prefetches string values from disk into an input queue.

        In training the capacity of the queue is important because a larger queue
        means better mixing of training examples between shards. The minimum number of
        values kept in the queue is values_per_shard * input_queue_capacity_factor,
        where input_queue_memory factor should be chosen to trade-off better mixing
        with memory usage.

        Args:
          reader: Instance of tf.ReaderBase.
          file_pattern: Comma-separated list of file patterns (e.g.
              /tmp/train_data-?????-of-00100).
          is_training: Boolean; whether prefetching for training or eval.
          batch_size: Model batch size used to determine queue capacity.
          values_per_shard: Approximate number of values per shard.
          input_queue_capacity_factor: Minimum number of values to keep in the queue
            in multiples of values_per_shard. See comments above.
          num_reader_threads: Number of reader threads to fill the queue.
          shard_queue_name: Name for the shards filename queue.
          value_queue_name: Name for the values input queue.

        Returns:
          A Queue containing prefetched string values.
        """
        data_files = []
        for filename in os.listdir(data_dir):
            filename = os.path.join(data_dir, filename)
            data_files.extend(tf.gfile.Glob(filename))
        if not data_files:
            tf.logging.fatal("Found no input files matching %s", data_dir)
        else:
            tf.logging.info("Prefetching values from %d files matching %s",
                            len(data_files), data_dir)

        if is_training:
            filename_queue = tf.train.string_input_producer(
                data_files, shuffle=True, capacity=16, name=shard_queue_name)
            min_queue_examples = values_per_shard * input_queue_capacity_factor
            capacity = min_queue_examples + 100 * batch_size
            values_queue = tf.RandomShuffleQueue(
                capacity=capacity,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string],
                name="random_" + value_queue_name)
        else:
            filename_queue = tf.train.string_input_producer(
                data_files, shuffle=False, capacity=1, name=shard_queue_name)
            capacity = values_per_shard + 3 * batch_size
            values_queue = tf.FIFOQueue(
                capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

        enqueue_ops = []
        for _ in range(num_reader_threads):
            _, value = reader.read(filename_queue)
            enqueue_ops.append(values_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
            values_queue, enqueue_ops))
        tf.summary.scalar(
            "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
            tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

        return values_queue

    def _build_data_inputs(self, data_dir):
        # Prefetch serialized SequenceExample protos.
        input_queue = self._prefetch_input_data(
            reader=self.tf_reader,
            data_dir=data_dir,
            is_training=True,
            batch_size=self.data_config.batch_size,
            values_per_shard=self.data_config.values_per_input_shard,
            input_queue_capacity_factor=self.data_config.input_queue_capacity_factor,
            num_reader_threads=self.data_config.num_input_reader_threads)

        assert self.data_config.num_preprocess_threads % 2 == 0
        images_and_captions = []
        for thread_id in range(self.data_config.num_preprocess_threads):
            serialized_sequence_example = input_queue.dequeue()
            image, caption = self._parse_sequence_example(serialized_sequence_example)
            images_and_captions.append([image, caption])

        # Batch inputs.
        queue_capacity = (2 * self.data_config.num_preprocess_threads *
                          self.data_config.batch_size)
        images, input_seqs, target_seqs, input_mask = (
            self._batch_with_dynamic_pad(images_and_captions,
                                         batch_size=self.data_config.batch_size,
                                         queue_capacity=queue_capacity))
        return images, input_seqs, target_seqs, input_mask


def main(_):
    """
    example for data_reader
    :return:
    """
    data_reader = ImageCaptionDataReader()
    inputs = data_reader._build_data_inputs(data_dir=data_reader.data_config.train_data_dir)
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for batch_index in range(5):
                data_batch = sess.run([inputs])
                for idx, data in enumerate(data_batch):
                    print("batch={}, data={}".format(batch_index, data))

        except Exception as e:
            print(e)
            coord.request_stop(e)
        finally:
            coord.request_stop()  # Stop the threads
            coord.join(threads)  # Wait for threads to stop


if __name__ == '__main__':
    tf.app.run()
