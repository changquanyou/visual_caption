# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec

from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader


class ImageCaptionDataEmbedding():
    def __init__(self):
        self.data_loader = ImageCaptionDataLoader()
        self.data_config = ImageCaptionDataConfig()
        self.load_embeddings()

    def build_char_text(self, json_data_file, image_dir):
        """
        build sentence file for embeddings

        example:    <S> 这 是 一 个 例 子 </S>
        :param json_data_file:
        :return:
        """
        print("begin char txt generation for {}".format(json_data_file))
        raw_data_gen = self.load_raw_generator(json_data_file=json_data_file, image_dir=image_dir)
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

    def build_char_all(self):
        """
        generate char txt file
            Each sentence in test and train data is tokenized to Chinese char in per line.
        :return:
        """
        print("begin char txt generation ")
        self.build_char_text(json_data_file=self.data_config.train_json_data,
                             image_dir=self.data_config.train_image_dir)
        self.build_char_text(json_data_file=self.data_config.validation_json_data,
                             image_dir=self.data_config.validation_image_dir)
        print("end char txt  generation")

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

    def load_embeddings(self):
        """
        load char2vec or word2vec model for token embeddings
        :return:
        """
        token2vec = Word2Vec.load(self.data_config.char2vec_model)
        vocab = token2vec.wv.vocab

        self.token2index = dict()
        self.index2token = dict()
        self.token_embedding_matrix = np.zeros([len(vocab)+1, self.data_config.embedding_dim_size])

        for idx, token in enumerate(token2vec.wv.index2word):
            token_embedding = token2vec.wv[token]
            self.index2token[idx] = token
            self.token2index[token] = idx
            self.token_embedding_matrix[idx] = token_embedding
        idx+=1
        # for unknown token
        self.token_embedding_matrix[idx] = np.zeros(shape=[self.data_config.embedding_dim_size])
        self.token2index[self.data_config.unknown_token] = len(vocab)
        self.index2token[idx] = self.data_config.unknown_token
        self.vocab_size = len(self.token2index)

        pass

    def text_to_ids(self, text):

        """
          not each token is not in token2index dict
          :param caption_txt:
          :return:
          """
        if not self.token2index:
            self.load_embeddings()

        ids = []
        for token in text:
            if token in self.token2index.keys():
                ids.append(self.token2index[token])
            else:
                ids.append(self.token2index[self.data_config.unknown_token])
        return ids
