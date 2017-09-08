# -*- coding:utf-8 -*-
import os
from gensim.models import Word2Vec


def train_embe(data_dir,out_dir):
    """
    训练data_dir下多个文件的不同维度的word2vec模型，并保存到out_dir下
    """
    files = os.listdir(data_dir)
    sents=[]
    for file in files:
        doc = os.path.join(data_dir,file)
        for line in open(doc,'r',encoding='utf-8'):
            sents.append(line.split("\t"))
    sizes=[50,100,200,300,500]
    for size in sizes:
        out_="model"+str(size)
        out_file = os.path.join(out_dir,out_)
        model = Word2Vec(
        sentences=sents, size=size, alpha=0.025, window=5, min_count=1,
        max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
        sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
        trim_rule=None, sorted_vocab=1
        )
        model.wv.save_word2vec_format(out_file, binary=False)


def get_vocb_dict(model_path):
    model = Word2Vec().wv.load_word2vec_format(model_path, binary=False)
    vocabulary_word2index = {}
    vocabulary_index2word = {}
    vocabulary_word2index['PAD_ID'] = 0
    vocabulary_index2word[0] = 'PAD_ID'
    special_index = 1
    for i, vocab in enumerate(model.vocab):
        vocabulary_word2index[vocab] = i + 1 + special_index
        vocabulary_index2word[i + 1 + special_index] = vocab
    return vocabulary_word2index,vocabulary_index2word

def train_data_to_id(train_data,vocabulary_word2index):
    train_id=[]
    for line in train_data:
        for word in line:
            train_id.append(vocabulary_word2index[word])
    return train_id

if __name__ == '__main__':
    data_dir = "C:/Users/tsf/Desktop/gitdata/visual_caption/data/tinyshakespeare"
    out_dir = "C:/Users/tsf/Desktop/gitdata/visual_caption/data/word2vec"
    train_embe(data_dir, out_dir)
