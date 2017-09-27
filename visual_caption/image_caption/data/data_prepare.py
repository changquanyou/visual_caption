# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
import sys

import ijson
import numpy as np
import tensorflow as tf
from gensim.models.word2vec import LineSentence, Word2Vec

from visual_caption.base.data.base_data_prepare import BaseDataPrepare
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


class ImageCaptionDataPrepare(BaseDataPrepare):
    """
    Prepare caption txt and token2vec model
    """

    def __init__(self):
        self.data_config = ImageCaptionDataConfig()
        # self.load_embeddings(self.data_config.embedding_dim_size)
        self.image_decoder = ImageDecoder()

    def build_char_all(self):
        """
        generate char txt file
            Each sentence in test and train data is tokenized to Chinese char in per line.
        :return:
        """
        print("begin char txt generation ")
        self.build_char_text(json_data_file=self.data_config.train_json_data)
        self.build_char_text(json_data_file=self.data_config.validation_json_data)
        print("end char txt  generation")

    def build_char_text(self, json_data_file):
        """
        build sentence file for embeddings

        example:    <S> 这 是 一 个 例 子 </S>
        :param json_data_file:
        :return:
        """
        print("begin char txt generation for {}".format(json_data_file))
        raw_data_gen = self.rawdata_generator(json_data_file=json_data_file)
        with open(file=self.data_config.caption_char_txt, mode='a', encoding='utf-8') as f_txt:
            for batch, batch_data in enumerate(raw_data_gen):
                for raw_data in batch_data:
                    captions = raw_data['captions']
                    for caption in captions:
                        if len(str.strip(caption)) > 0:
                            line = [char + ' ' for char in caption]  # separate each token with a whitespace
                            line.insert(0, self.data_config.begin_token + " ")
                            line.append(self.data_config.end_token)
                            line.append('\n')
                            f_txt.writelines(line)
                if batch % 1000 == 0:
                    print("Generating caption char txt for batch={}".format(batch * 1000))
            print("end char txt generation for {}".format(json_data_file))

        pass

    def rawdata_generator(self, json_data_file):
        """
        load json file and yield data in batch
        :param json_data_file:
        :param batch_size:
        :return:
        """
        batch_data = []
        count = 0;
        with open(json_data_file, mode='r', encoding='utf-8') as f_json:
            item_gen = ijson.items(f_json, "item")
            for item in enumerate(item_gen):
                (id, caption_dict) = item
                url = caption_dict['url']
                image_id = caption_dict['image_id']
                captions = caption_dict['caption']
                caption_list = []
                for idx, caption_txt in enumerate(captions):
                    caption_txt = str.strip(caption_txt)
                    caption_txt = caption_txt.replace(' ', '')
                    caption_txt = caption_txt.replace('\n', '')
                    caption_list.append(caption_txt)
                caption_data = {'id': id, 'url': url, 'image_id': image_id, 'captions': caption_list}
                batch_data.append(caption_data)
                if len(batch_data) == self.data_config.batch_size:
                    yield batch_data
                    batch_data = []

                count += 1
                if count % 1000 == 0:
                    print("load {} image data instances".format(count))

            if len(batch_data) > 0:
                yield batch_data
                del batch_data

    def build_embeddings(self):
        sentences = LineSentence(self.data_config.caption_char_txt)
        dims = [50, 100, 200, 300]
        for dim_size in dims:
            model_file_name = "char2vec_" + str(dim_size) + ".model"
            model_file = os.path.join(self.data_config.embedding_dir, model_file_name)
            print("begin token2vec model {} generation".format(model_file))
            model = Word2Vec(sentences, size=dim_size, window=5, min_count=1, workers=4)
            model.save(model_file)
            print("Generated token2vec model to {}".format(model_file))

    def generate_word_file(self):
        pass

    def load_embeddings(self):
        """
        load char2vec or word2vec model for token embeddings
        :return:
        """
        if not os.path.isfile(self.data_config.char2vec_model):
            self.build_embeddings()

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
        self.token2index[self.data_config.unknown_token] = len(self.vocab)  # for unknown token
        pass

    def build_tf_data(self, mode):
        """
        Converts AI_Challenge Image_Caption data to TFRecord file format with SequenceExample protos.

        """

        if mode == 'train':
            image_dir = self.data_config.train_image_dir
            caption_gen = self.rawdata_generator(self.data_config.train_json_data)
            output_file = self.data_config.train_tf_data_file
        elif mode == 'test':
            image_dir = self.data_config.test_image_dir
            caption_gen = self.rawdata_generator(self.data_config.test_json_data)
            output_file = self.data_config.test_tf_data_file
        elif mode == 'validation':
            image_dir = self.data_config.validation_image_dir
            caption_gen = self.rawdata_generator(self.data_config.validation_json_data)
            output_file = self.data_config.validation_tf_data_file

        writer = tf.python_io.TFRecordWriter(output_file)

        for batch, batch_data in enumerate(caption_gen):
            for caption_data in batch_data:
                sequence_example_list = self._to_sequence_example(image_dir=image_dir, caption_data=caption_data)
                for sequence_example in sequence_example_list:
                    if sequence_example is not None:
                        writer.write(sequence_example.SerializeToString())
            if batch % 100 == 0 and batch > 0:
                print("flush batch {} dataset into file {}".format(batch, output_file))

            sys.stdout.flush()
            if batch % 1000 == 0 and batch > 0:
                break

        writer.close()
        sys.stdout.flush()

    def caption_to_ids(self, caption_txt):
        """
        not each token is not in token2index dict
        :param caption_txt:
        :return:
        """
        ids = [self.token2index[token] for token in caption_txt]
        return ids

    def _to_sequence_example(self, image_dir, caption_data):
        id = caption_data['id']  # instance id

        image_id = caption_data['image_id']
        encoded_image_id = image_id.encode()
        image_filename = os.path.join(image_dir, image_id)
        with tf.gfile.FastGFile(image_filename, "rb") as f:
            encoded_image = f.read()

        try:
            self.image_decoder.decode_jpeg(encoded_image)
        except (tf.errors.InvalidArgumentError, AssertionError):
            print("Skipping file with invalid JPEG data: %s" % image_filename)
            return
        # shape = encoded_image.shape

        url = caption_data['url']
        caption_list = caption_data['captions']

        context = tf.train.Features(feature={
            "image/image_id": self._bytes_feature(encoded_image_id),
            "image/rawdata": self._bytes_feature(encoded_image),
        })

        sequence_example_list = []
        for caption_txt in caption_list:
            encoded_caption_txt = [token.encode() for token in caption_txt.split()]
            caption_ids = self.caption_to_ids(caption_txt)
            feature_lists = tf.train.FeatureLists(feature_list={
                "image/caption": self._bytes_feature_list(encoded_caption_txt),
                "image/caption_ids": self._int64_feature_list(caption_ids)
            })
            sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
            sequence_example_list.append(sequence_example)

        return sequence_example_list


if __name__ == '__main__':
    data_prepare = ImageCaptionDataPrepare()

    # build sentence file for embeddings
    # data_prepare.build_char_all()

    # build embeddings models
    # data_prepare.build_embeddings()

    # load embeddings models
    data_prepare.load_embeddings()

    modes = ['train', 'validation']
    for mode in modes:
        data_prepare.build_tf_data(mode=mode)
        # data_prepare.generate_char_txt()
        # data_prepare.generate_char2vec()
