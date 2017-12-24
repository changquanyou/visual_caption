# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import numpy as np
import tensorflow as tf
from gensim.models.word2vec import LineSentence, Word2Vec
from tensorflow.contrib.tensorboard.plugins import projector

from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.image_caption.data.data_loader import ImageCaptionDataLoader
from visual_caption.utils.decorator_utils import timeit


class ImageCaptionDataEmbedding(object):
    def __init__(self):
        self.data_loader = ImageCaptionDataLoader()
        self.data_config = ImageCaptionDataConfig()
        self.load_embeddings()


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

    @timeit
    def load_embeddings(self):
        """
        load char2vec or word2vec model for token embeddings
        :return:
        """
        token2vec = Word2Vec.load(self.data_config.char2vec_model)
        self.vocab = dict()
        for token, item in token2vec.wv.vocab.items():
            self.vocab[token] = {'count': item.count,
                                 'index': item.index}
        self.vocab[self.data_config.token_unknown] = {'count': 0,
                                                      'index': len(token2vec.wv.vocab)}

        self.token2index = dict()
        self.index2token = dict()
        self.token_embedding_matrix = np.zeros([len(self.vocab),
                                                self.data_config.embedding_dim_size])

        for idx, token in enumerate(token2vec.wv.index2word):
            token_embedding = token2vec.wv[token]
            self.index2token[idx] = token
            self.token2index[token] = idx
            self.token_embedding_matrix[idx] = token_embedding
        idx += 1
        # for unknown token
        self.token_embedding_matrix[idx] = np.zeros(shape=[self.data_config.embedding_dim_size])
        self.token2index[self.data_config.token_unknown] = idx
        self.index2token[idx] = self.data_config.token_unknown

        self.vocab_size = len(self.vocab)
        self.embedding_size = self.data_config.embedding_dim_size
        pass

    def visualize(self, model):
        if not model:
            model = Word2Vec.load(self.data_config.char2vec_model)
        meta_file = os.path.join(self.data_config.embedding_dir, "metadata.tsv")
        placeholder = np.zeros((len(model.wv.index2word), self.data_config.embedding_dim_size))
        with open(meta_file, 'wb') as file_metadata:
            for i, word in enumerate(model.wv.index2word):
                placeholder[i] = model[word]
                # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
                if word == '':
                    print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                    file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
                else:
                    file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

                    # define the model without training
        sess = tf.InteractiveSession()
        token_embeddings = tf.Variable(placeholder, trainable=False, name='token_embeddings')
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(self.data_config.embedding_dir, sess.graph)

        # adding into projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'token_embeddings'
        embed.metadata_path = meta_file

        # Specify the width and height of a single thumbnail.
        projector.visualize_embeddings(writer, config)
        saver.save(sess, os.path.join(self.data_config.embedding_dir, 'metadata.ckpt'))
        print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(
            self.data_config.embedding_dir))

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
                ids.append(self.token2index[self.data_config.token_unknown])
        return ids


if __name__ == '__main__':
    data_embeddings = ImageCaptionDataEmbedding()
    data_embeddings.build_char_all()
    data_embeddings.build_embeddings()
