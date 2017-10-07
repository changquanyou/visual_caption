# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os

import ijson.backends.yajl2_cffi as ijson

from gensim.models.word2vec import LineSentence, Word2Vec

from visual_caption.base.data.base_data_loader import BaseDataLoader
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig

default_data_config = ImageCaptionDataConfig()

import tensorflow as tf
import numpy as np


class ImageCaptionDataLoader(BaseDataLoader):
    """

    Data loader of raw data and prepare data for preprocessing

    Data Preprocessing:

        Prepare caption txt and token2vec model
    """

    def __init__(self, data_config=default_data_config):
        super().__init__(data_config=data_config)
        self.load_embeddings()

    def load_raw_generator(self, json_data_file, image_dir):
        """
        load json file and yield data in batch
        :param json_data_file:
        :param batch_size:
        :return:
        """
        batch_data = []

        # load_batch_size = self.data_config.batch_size
        load_batch_size = 80
        # count = 0;
        with open(json_data_file, mode='rb') as f_json:
            item_gen = ijson.items(f_json, "item")
            for item in enumerate(item_gen):
                (id, caption_dict) = item
                url = caption_dict['url']
                image_id = caption_dict['image_id']
                image_file = os.path.join(image_dir, image_id)
                captions = caption_dict['caption']
                caption_list = []
                for idx, caption_txt in enumerate(captions):
                    caption_txt = str.strip(caption_txt)
                    caption_txt = caption_txt.replace(' ', '')
                    caption_txt = caption_txt.replace('\n', '')
                    caption_list.append(caption_txt)

                caption_data = {
                    'id': id,
                    'url': url,
                    'image_id': image_id,
                    'image_file': image_file,
                    'captions': caption_list
                }
                batch_data.append(caption_data)
                if len(batch_data) == load_batch_size:
                    yield batch_data
                    batch_data = []

                # count += 1
                # if count % 1000 == 0:
                #     print("load {} image data instances".format(count))

            if len(batch_data) > 0:
                yield batch_data
                del batch_data

    def load_test_data(self):
        return self.load_raw_generator(json_data_file=self.data_config.test_json_data,
                                       image_dir=self.data_config.test_image_dir)

    def load_train_data(self):
        return self.load_raw_generator(json_data_file=self.data_config.train_json_data,
                                       image_dir=self.data_config.train_image_dir)

    def load_validation_data(self):
        return self.load_raw_generator(json_data_file=self.data_config.validation_json_data,
                                       image_dir=self.data_config.validation_image_dir)

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


def main(_):
    data_loader = ImageCaptionDataLoader()


if __name__ == '__main__':
    tf.app.run()
