import codecs

import numpy as np
import os
from gensim.models import Word2Vec


class TextLoader():
    def __init__(self,word2vec_model, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.word2vec_model=word2vec_model
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        self.preprocess(input_file)

        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file):
        with codecs.open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        word2index, index2word = self.get_vocb_dict(self.word2vec_model)
        datas = []
        for line in lines:
            words = line.split()
            for word in words:
                datas.append(word2index[word])
        self.tensor = np.array(datas)

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

    def get_vocb_dict(self, model_path):
        model = Word2Vec().wv.load_word2vec_format(model_path, binary=False)
        vocabulary_word2index = {}
        vocabulary_index2word = {}
        special_index = 1
        for i, vocab in enumerate(model.index2word):
            vocabulary_word2index[vocab] = i
            vocabulary_index2word[i] = vocab
        return vocabulary_word2index, vocabulary_index2word

if __name__ == '__main__':


    batch_size = 20
    seq_length = 100
    data_dir = "/home/liuxiaoming/data/tf-visgen/TextGeneration/data"
    word2vec_model="word2vec/model100"
    loader = TextLoader(word2vec_model=word2vec_model,data_dir=data_dir, batch_size=batch_size, seq_length=seq_length, encoding="utf-8")
    batch = loader.next_batch()
    while batch:
        print("batch_x={}, batch_y=".format(batch[0],batch[1]))
        batch = loader.next_batch()
