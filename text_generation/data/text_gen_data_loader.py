# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging

from gensim.models.keyedvectors import KeyedVectors

from visual_caption.base.data.base_data_loader import BaseDataLoader
from text_generation.data.text_gen_data_config import TextGenDataConfig
from text_generation.data.utils import TextLoader

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
        batch_size = self.data_config.batch_size
        seq_length = 100
        data_dir = "C:/Users/tsf/Desktop/gitdata/visual_caption/data/tinyshakespeare"
        word2vec_model = self.data_config.toke2vec_file
        loader = TextLoader(word2vec_model=word2vec_model,data_dir=data_dir, batch_size=batch_size, seq_length=seq_length, encoding="utf-8")
        for b in range(loader.num_batches):
            data_batch=loader.next_batch()
            yield data_batch


    def load_test_data(self):
        pass

    def load_infer_data(self):
        pass

    pass


if __name__ == '__main__':
    data_config = TextGenDataConfig(model_name="TextGeneration")
    data_loader = TextGenDataLoader(data_config=data_config)
    train_gen = data_loader.load_train_data()
