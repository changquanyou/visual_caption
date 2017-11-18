# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import logging
import os
from collections import defaultdict
from pathlib import Path
from pprint import pprint  # pretty-printer

import numpy  as np
from gensim import corpora
from gensim.models.word2vec import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

home = str(Path.home())
caption_dir = os.path.join(home, 'data/ai_challenge/image_caption/captions')
caption_char_file = os.path.join(caption_dir, 'caption_char.txt')
embedding_dir = os.path.join(home, 'data/ai_challenge/image_caption/embeddings')


class Embedding_Test(object):
    def __init__(self):
        self.char2vec_model = os.path.join(embedding_dir, 'char2vec_100.model')
        self.load_embeddings()

    def load_embeddings(self):
        """
        load char2vec or word2vec model for token embeddings
        :return:
        """
        self.token2vec = Word2Vec.load(self.char2vec_model)
        self.vocab = self.token2vec.wv.vocab

        self.token2index = {}
        self.index2token = {}
        self.token_embedding_matrix = np.zeros([len(self.vocab) + 1, 100])
        for idx, token in enumerate(self.token2vec.wv.index2word):
            token_embedding = self.token2vec.wv[token]
            self.index2token[idx] = token
            self.token2index[token] = idx
            self.token_embedding_matrix[idx] = token_embedding

        # for unknown token
        self.token2index['UNKNOWN'] = len(self.vocab)
        self.vocab_size = len(self.token2index)

        pass

    def test_embeddings(self):
        with open(file=caption_char_file, mode='r', encoding='utf-8') as f:
            sentences = f.readlines()
            for idx, sentence in enumerate(sentences):
                sentence = str.strip(sentence)
                ids = [self.token2index[char] for char in sentence.split()]
                print('sentence[{}] = {}'.format(idx, sentence))
                print('ids[{}] = {}\n'.format(idx, ids))


def build_vocab(filename):
    with open(file=filename, mode='r', encoding='utf-8') as f:
        sentences = f.readlines()
        texts = [[char for char in sentence.split()]
                 for sentence in sentences]

    # frequency = defaultdict(int)
    # for text in texts:
    #     for token in text:
    #         frequency[token] += 1

    # texts = [[token for token in text if frequency[token] >= 1]
    #          for text in texts]
    dictionary = corpora.Dictionary(texts)
    vocabl_char_file = os.path.join(embedding_dir, 'vocab_char.txt')
    dictionary.save_as_text(fname=vocabl_char_file)

    # dictionary = corpora.Dictionary().load_from_text(vocabl_char_file)

    index2token = dictionary.id2token
    token2index = dictionary.token2id

    # dictionary.save_as_text(dict_txt_file)

    # dictionary = dictionary.load_from_text(fname=caption_char_file)

    # char_dict_file = os.path.join(caption_dir, 'caption_char.dict')
    # dictionary.save(char_dict_file)  # store the dictionary, for future reference
    # pprint(dictionary)


if __name__ == '__main__':
    build_vocab(filename=caption_char_file)
