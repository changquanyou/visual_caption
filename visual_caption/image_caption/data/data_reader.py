# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import tensorflow as tf
from gensim import corpora
from tensorflow.python.ops import lookup_ops

from visual_caption.base.data.base_data_reader import BaseDataReader
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
# Data Reader class for AI_Challenge_2017
from visual_caption.image_caption.data.data_embedding import ImageCaptionDataEmbedding


class ImageCaptionDataReader(BaseDataReader):
    """
        Read data for train, validation, test dataset with embedding model
    """
    def __init__(self, data_config):
        self._data_config = data_config
        self.data_embedding = ImageCaptionDataEmbedding()
        self._vocab_table = lookup_ops.index_table_from_file(
            vocabulary_file=self._data_config.vocab_file, default_value=0)
        super(ImageCaptionDataReader, self).__init__(
            data_config=data_config)

    def _build_context_and_feature(self):
        self.context_features = {
            self._data_config.visual_image_id_name:
                tf.FixedLenFeature([], dtype=tf.string),
            self._data_config.visual_feature_name:
                tf.FixedLenFeature([], dtype=tf.string),
        }
        self.sequence_features = {
            self._data_config.caption_text_name:
                tf.FixedLenSequenceFeature([], dtype=tf.string),
            # self._data_config.caption_ids_name:
            # tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

    def _mapping_dataset(self, dataset):
        num_threads = self._data_config.num_preprocess_threads
        output_buffer_size = self._data_config.output_buffer_size
        random_seed = self._data_config.random_seed

        dataset = dataset.shuffle(buffer_size=output_buffer_size,
                                  seed=random_seed)

        # Create a phrase prefixed with <sos> and create a target suffixed with <eos>.
        token_begin = self._data_config.token_begin
        token_end = self._data_config.token_end

        # add target
        dataset = dataset.map(
            lambda image_id, image_feature, caption: (
                image_id, image_feature,
                tf.concat(([token_begin], caption), axis=0),  # modify phase to begin_token  +  phrase
                tf.concat((caption, [token_end]), axis=0)  # create target = phrase + end_token
            ),
            num_parallel_calls=num_threads)

        # add caption_ids and target_ids
        dataset = dataset.map(
            lambda image_id, image_feature, caption, target: (
                image_id, image_feature, caption, target,
                tf.cast(self._vocab_table.lookup(caption), tf.int32),
                tf.cast(self._vocab_table.lookup(target), tf.int32)
            ),
            num_parallel_calls=num_threads)

        # Add in sequence lengths.
        dataset = dataset.map(
            lambda image_id, image_feature, caption, target, caption_ids, target_ids: (
                image_id, image_feature, caption, target, caption_ids, target_ids,
                tf.size(caption_ids),  # caption_length
                tf.size(target_ids)  # target_length
            ),
            num_parallel_calls=num_threads)

        def batching_func(x):
            return x.padded_batch(batch_size=self._batch_size,
                                  padded_shapes=(
                                      tf.TensorShape([]),  # image_id
                                      tf.TensorShape([self._data_config.visual_feature_size]),  # image_feature

                                      tf.TensorShape([None]),  # caption
                                      tf.TensorShape([None]),  # target

                                      tf.TensorShape([None]),  # caption_ids
                                      tf.TensorShape([None]),  # target_ids

                                      tf.TensorShape([]),  # phrase_length
                                      tf.TensorShape([]),  # target_length
                                  )
                                  )

        batched_dataset = batching_func(dataset)
        return batched_dataset

    def _parse_tf_example(self, serialized_example):
        """
        parsing a tf example to data structrue
        :param serialized_example:
        :return:
        """
        # parsing sequence example
        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features=self.context_features,
            sequence_features=self.sequence_features
        )

        image_id = context[self._data_config.visual_image_id_name]
        visual_feature = context[self._data_config.visual_feature_name]

        # for vgg19 fc7
        visual_feature = tf.decode_raw(visual_feature, tf.float32)
        image_feature = tf.reshape(visual_feature, [self._data_config.visual_feature_size])

        caption = sequence[self._data_config.caption_text_name]
        # caption_ids = sequence[self._data_config.caption_ids_name]
        data = (image_id, image_feature, caption)
        return data

class ImageCaptionAttentionDataReader(BaseDataReader):
    def __init__(self,data_config):
        super(ImageCaptionAttentionDataReader,self).__init__(
            data_config = data_config
        )
        vocab_char_file = self._data_config.vocab_char_file
        dictionary = corpora.Dictionary().load_from_text(vocab_char_file)
        self.index2token = dictionary.id2token
        self.token2index = dictionary.token2id
    pass

def main(_):
    """
    example for data_reader
    :return:
    """
    global_step_tensor = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES]
    )
    data_reader = ImageCaptionDataReader()
    next_batch = data_reader.get_next_batch(batch_size=20)
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    train_init_op = data_reader.get_train_init_op()
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(tf.tables_initializer())
        sess.run(train_init_op)
        global_step = tf.train.global_step(sess, global_step_tensor)
        while True:
            try:
                batch_data = sess.run(next_batch)
                (image_ids, image_features, captions, targets,
                 caption_ids, target_ids, caption_lengths, target_lengths) = batch_data
                for idx, feature in enumerate(image_features):
                    caption = b' '.join(captions[idx])
                    caption = caption.decode()

                    target = b' '.join(targets[idx])
                    target = target.decode()

                    print("idx={}, image_id={}, \ncaption=[{}]\ncaption_id=[{}]\ntarget=[{}]\ntarget_id=[{}]"
                          .format(idx, image_ids[idx], caption, caption_ids[idx], target, target_ids[idx]))
                global_step = tf.train.global_step(sess, global_step_tensor)
            except tf.errors.OutOfRangeError:  # ==> "End of validation dataset"
                print("data reader finished at step={0}".format(global_step))
                break


if __name__ == '__main__':
    tf.app.run()
