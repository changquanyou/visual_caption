# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging

from gensim.models.keyedvectors import KeyedVectors

from tf_visgen.base.data.base_data_loader import BaseDataLoader
from tf_visgen.text_gen.data.text_gen_data_config import TextGenDataConfig

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TextGenDataLoader(BaseDataLoader):
    def __init__(self, data_config):
        super().__init__(data_config=data_config)
        self.tag_num = 200
        self._load_embeddings()

    def _load_embeddings(self):
        print("begin loading embedding")
        w2v_model = KeyedVectors.load_word2vec_format(self.data_config.toke2vec_file,
                                                      binary=True)
        self.vocab_size = len(w2v_model.index2word)
        self.embedding_dim = w2v_model.vector_size
        self.token_embedding_matrix = w2v_model.syn0
        print("end loading embedding")

    def load_train_data(self):



        pass

    def load_test_data(self):
        pass

    def load_infer_data(self):
        pass

    pass


if __name__ == '__main__':
    data_config = TextGenDataConfig(model_name="TextGeneration")
    data_loader = TextGenDataLoader(data_config=data_config)
    train_gen = data_loader.load_train_data()
