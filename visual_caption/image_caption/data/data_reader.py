# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

# Data Reader class for AI_Challenge_2017
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.contrib.learn import ModeKeys
from visual_caption.base.data.base_data_reader import BaseDataReader
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig


class Vocabulary(object):
    """Vocabulary class for an image-to-text model."""

    def __init__(self,
                 vocab_file,
                 start_word="<S>",
                 end_word="</S>",
                 unk_word="<UNK>"):
        """Initializes the vocabulary.

        Args:
          vocab_file: File containing the vocabulary, where the words are the first
            whitespace-separated token on each line (other tokens are ignored) and
            the word ids are the corresponding line numbers.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
        """
        if not tf.gfile.Exists(vocab_file):
            tf.logging.fatal("Vocab file %s not found.", vocab_file)
        tf.logging.info("Initializing vocabulary from file: %s", vocab_file)

        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab]
        assert start_word in reverse_vocab
        assert end_word in reverse_vocab
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

        tf.logging.info("Created vocabulary with %d words" % len(vocab))

        self.vocab = vocab  # vocab[word] = id
        self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

        # Save special word ids.
        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = vocab[unk_word]
        self.num_vocab = len(self.vocab)

    def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        else:
            return self.reverse_vocab[word_id]


class ImageCaptionDataReader(BaseDataReader):
    """
      read tf_example from mscoco tfrecord file ,
      convert tf_example data into tf.data.Dataset format
      """

    def __init__(self, data_config):
        self.vocab_table = lookup_ops.index_table_from_file(
            vocabulary_file=data_config.vocab_file,
            default_value=0)
        self.vocabulary = Vocabulary(vocab_file=data_config.vocab_file,
                                     start_word=data_config.token_start,
                                     end_word=data_config.token_end,
                                     unk_word=data_config.token_unknown)
        super(ImageCaptionDataReader, self).__init__(
            data_config=data_config)

    def _mapping_dataset(self, dataset):
        num_threads = self.data_config.num_preprocess_threads
        dim_visual_feature = self.data_config.dim_visual_feature
        buffer_size = self.data_config.output_buffer_size
        random_seed = self.data_config.random_seed
        if self.data_config.mode == ModeKeys.TRAIN:
            dataset = dataset.shuffle(
                buffer_size=buffer_size, seed=random_seed)

        # add target and add prefix to caption and suffix to
        token_start = self.data_config.token_start
        token_end = self.data_config.token_end
        token_pad = self.data_config.token_pad
        token_pad_id = self.vocabulary.vocab[token_pad]

        dataset = dataset.map(
            lambda image_id, image_height, image_width, image_depth, image_feature,
                   bbox_features_shape, bbox_number, bbox_labels, bboxes, bbox_features,
                   caption: (
                image_id, image_height, image_width, image_depth, image_feature,
                bbox_features_shape, bbox_number, bbox_labels, bboxes, bbox_features,
                # tf.slice(caption, begin=[], size=[0, ])

                tf.concat(([token_start], caption), axis=0),
                tf.concat((caption, [token_end]), axis=0)

            ), num_parallel_calls=num_threads)

        # add caption_ids and target_ids
        dataset = dataset.map(
            lambda image_id, image_height, image_width, image_depth, image_feature,
                   bbox_features_shape, bbox_number, bbox_labels, bboxes, bbox_features,
                   caption, target: (
                image_id, image_height, image_width, image_depth, image_feature,
                bbox_features_shape, bbox_number, bbox_labels, bboxes, bbox_features,
                caption, target,
                tf.cast(self.vocab_table.lookup(caption), tf.int32),
                tf.cast(self.vocab_table.lookup(target), tf.int32)
            ), num_parallel_calls=num_threads)

        # Add in sequence lengths.
        dataset = dataset.map(
            lambda image_id, image_height, image_width, image_depth, image_feature,
                   bbox_features_shape, bbox_number, bbox_labels, bboxes, bbox_features,
                   caption, target, caption_ids, target_ids: (

                image_id, image_height, image_width, image_depth, image_feature,
                bbox_features_shape, bbox_number, bbox_labels, bboxes, bbox_features,
                caption, target, caption_ids, target_ids,
                tf.size(caption_ids),  # caption_length
                tf.size(target_ids)  # target_length
            ), num_parallel_calls=num_threads)

        def batching_func(x):
            return x.padded_batch(
                batch_size=self.data_config.batch_size,
                padded_shapes=(
                    tf.TensorShape([]),  # image_id
                    tf.TensorShape([]),  # width
                    tf.TensorShape([]),  # height
                    tf.TensorShape([]),  # depth
                    tf.TensorShape([dim_visual_feature]),  # image_feature

                    tf.TensorShape([None]),  # image_bbox_shape
                    tf.TensorShape([]),  # number of bboxes
                    tf.TensorShape([None]),  # labels
                    tf.TensorShape([None]),  # bboxes
                    tf.TensorShape([None, None]),  # image_bbox_features

                    tf.TensorShape([None]),  # caption
                    tf.TensorShape([None]),  # target
                    tf.TensorShape([None]),  # caption_ids
                    tf.TensorShape([None]),  # target_ids

                    tf.TensorShape([]),  # phrase_length
                    tf.TensorShape([]),  # target_length
                ),
                padding_values=(
                    token_pad, np.int32(0), np.int32(0), np.int32(0), np.float32(0),
                    np.int32(0), np.int32(0), np.int64(0), np.int64(0), np.float32(0),

                    token_pad, token_pad,
                    np.int32(token_pad_id), np.int32(token_pad_id),
                    np.int32(0), np.int32(0)
                )

            )

        dataset = batching_func(dataset)
        return dataset
        pass

    def _build_context_and_feature(self):
        self.context_features = {
            'image/image_id': tf.FixedLenFeature([], dtype=tf.string),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64),
            'image/depth': tf.FixedLenFeature([], dtype=tf.int64),
            'image/feature': tf.FixedLenFeature([], dtype=tf.string),

            'bbox/number': tf.FixedLenFeature([], dtype=tf.int64),
            'bbox/labels': tf.VarLenFeature(tf.int64),
            'bbox/bboxes': tf.VarLenFeature(tf.int64),
            'bbox/features': tf.FixedLenFeature([], dtype=tf.string),
        }
        self.sequence_features = {
            'caption': tf.FixedLenSequenceFeature([], dtype=tf.string),
        }
        pass

    def _parse_tf_example(self, serialized_example):
        # parsing sequence example
        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features=self.context_features,
            sequence_features=self.sequence_features
        )

        image_id = context['image/image_id']
        image_height = tf.cast(context['image/height'], tf.int32)
        image_width = tf.cast(context['image/width'], tf.int32)
        image_depth = tf.cast(context['image/depth'], tf.int32)

        # image_rawdata = tf.decode_raw(context['image/raw_data'], tf.float32)
        # image_shape = tf.stack([image_height, image_width, image_depth])
        # image_rawdata = tf.reshape(image_rawdata, image_shape)

        image_feature = tf.decode_raw(context['image/feature'], tf.float32)

        # # Since information about shape is lost reshape it
        bbox_number = tf.cast(context['bbox/number'], tf.int32)
        bbox_labels = tf.sparse_tensor_to_dense(context['bbox/labels'], default_value=0)
        bboxes = tf.sparse_tensor_to_dense(context['bbox/bboxes'], default_value=0)
        # bboxes_shape = tf.stack([bbox_number, 4])
        # bboxes = tf.reshape(bboxes, bboxes_shape)

        bbox_features = context['bbox/features']
        bbox_features = tf.decode_raw(bbox_features, tf.float32)
        # bbox_features = tf.sparse_tensor_to_dense(context['bbox/features'], default_value=0)
        bbox_features_shape = tf.stack([bbox_number, self.data_config.dim_visual_feature])
        bbox_features = tf.reshape(bbox_features, bbox_features_shape)

        caption = sequence["caption"]
        parsed_example = (image_id, image_height, image_width, image_depth, image_feature,
                          bbox_features_shape, bbox_number, bbox_labels, bboxes, bbox_features,
                          caption)
        return parsed_example
        pass

    pass


def print_output(batch_data):
    id_batch, width_batch, height_batch, depth_batch, feature_batch, \
    bbox_shape_batch, bbox_num, bbox_labels, bboxes, bbox_features, \
    caption_batch, target_batch, caption_ids, target_ids, caption_lengths, target_lengths = batch_data
    for idx, image_id in enumerate(id_batch):
        print("image: image_id={0:}, width={1:4d}, height={2:4d}, feature_shape={3:4d}"
              "\n\tbbox: features_shape={4:}, num={5:2d}, labels={6:}, bboxes={7:}, features={8:}"
              "\n\tcaption=[{9:}], \n\ttarget=[{10:}], caption_length={11:2d}, target_length={12:2d}".
              format(image_id, width_batch[idx], height_batch[idx], len(feature_batch[idx]),
                     bbox_shape_batch[idx], bbox_num[idx], len(bbox_labels[idx]),
                     len(bboxes[idx]) // 4, len(bbox_features[idx]),
                     str.strip("".join([token.decode() for token in caption_batch[idx]])),
                     str.strip("".join([token.decode() for token in target_batch[idx]])),
                     caption_lengths[idx], target_lengths[idx]))

    return batch_data


def main(_):
    data_config = ImageCaptionDataConfig()
    data_reader = ImageCaptionDataReader(data_config=data_config)

    next_batch = data_reader.get_next_batch(batch_size=10)
    dataset_init_op = data_reader.get_train_init_op()
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(dataset_init_op)
        step = 0
        while True:
            try:
                batch_data = sess.run(next_batch)
                print_output(batch_data)
                step += 1
            except tf.errors.OutOfRangeError:  # ==> "End of validation dataset"
                print("data reader finished at step={0}".format(step))
                break


if __name__ == '__main__':
    tf.app.run()
