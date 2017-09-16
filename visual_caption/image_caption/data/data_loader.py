# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import ijson
import numpy as np
import tensorflow as tf
from gensim.models.word2vec import Word2Vec

from visual_caption.base.data.base_data_loader import BaseDataLoader
from visual_caption.image_caption.data import data_utils
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig


# Data Loader class for AI_Challenge_2017

class ImageCaptionDataLoader(BaseDataLoader):
    """
        load token2vec model and data set for train, validation and test

    """

    def __init__(self, data_config):
        super().__init__(data_config=data_config)
        self._load_embeddings()

    def _load_embeddings(self):
        """
        load token2vec embeddings
        :return:
        """
        w2v_model = Word2Vec.load(self.data_config.char2vec_model)
        self.vocab = w2v_model.wv.vocab
        vocab_size = len(w2v_model.wv.index2word)  # initial vocab_size

        self.embedding_dim = w2v_model.vector_size

        self.token_embedding_matrix = np.zeros([vocab_size + 3, self.embedding_dim])

        self.word2index = {}
        self.index2word = {}
        for idx, word in enumerate(w2v_model.wv.index2word):
            word_embedding = w2v_model.wv[word]
            self.index2word[idx] = word
            self.word2index[word] = idx
            self.token_embedding_matrix[idx] = word_embedding

        self.index2word[vocab_size] = '#BEGIN#'
        self.index2word[vocab_size + 1] = '#END#'
        self.index2word[vocab_size + 2] = '#UNKNOWN#'

        self.word2index['#BEGIN#'] = vocab_size
        self.word2index['#END#'] = vocab_size + 1
        self.word2index['#UNKNOWN#'] = vocab_size + 2

        self.token_embedding_matrix[vocab_size] = np.zeros([self.embedding_dim])
        self.token_embedding_matrix[vocab_size + 1] = np.ones([self.embedding_dim])
        self.token_embedding_matrix[vocab_size + 2] = np.zeros([self.embedding_dim])-1

        self.vocab_size = vocab_size + 2  # +1 for #BEGIN#; +2 for #END#

    def _load_data(self, json_file, image_dir):
        """
        General data loading method
        :param json_file:
        :param image_dir:
        :return:
        """

        batch_data = []
        with open(json_file, mode='r', encoding='utf-8') as f:
            item_gen = ijson.items(f, "item")
            for item in enumerate(item_gen):
                (id, caption_dict) = item
                image_id = caption_dict['image_id']
                url = caption_dict['url']
                captions = caption_dict['caption']

                id_seqs = data_utils.convert_id_seqs(captions=captions, char2id=self.word2index)
                seqs, seq_lengths = data_utils.pad_sequences(id_seqs, 0, self.data_config.seq_max_length)

                for idx, seq in enumerate(seqs):
                    data = [id, image_id, url, seq, seq_lengths[idx]]
                    batch_data.append(data)
                    if len(batch_data) == self.data_config.batch_size:
                        yield batch_data
                        batch_data = []
            if len(batch_data) > 0:
                yield batch_data
        del batch_data

    def load_train_data(self):
        return self._load_data(json_file=self.data_config.train_json_data,
                               image_dir=self.data_config.train_image_dir)

    def load_test_data(self):
        return self._load_data(json_file=self.data_config.train_json_data)

    def load_validation_data(self):
        return self._load_data(json_file=self.data_config.validation_json_data,
                               image_dir=self.data_config.validation_image_dir)

    pass


def main(_):
    data_config = ImageCaptionDataConfig()
    data_loader = ImageCaptionDataLoader(data_config=data_config)
    data_gen = data_loader.load_validation_data()
    for batch, batch_data in enumerate(data_gen):
        print("batch={}".format(batch))
        for data in batch_data:
            print("data={}".format(data))


if __name__ == '__main__':
    tf.app.run()
