import os

from gensim.models.word2vec import Word2Vec

from tf_visgen.visgen.config import VisgenDataConfig
from tf_visgen.visgen.data.data_loader import DataLoader


class Word2VectorUtils(object):
    def __init__(self):
        self.visgen_config = VisgenDataConfig()
        self.visgen_loader = DataLoader(config=self.visgen_config)

    def prepare_phrase(self):
        region_gen = self.visgen_loader.load_regions()
        with open(self.visgen_config.region_phrase_txt, "a") as f:
            for batch_idx, region_batch in enumerate(region_gen):
                for idx, region in enumerate(region_batch):
                    f.write(region.phrase + "\n")

    def build_w2v(self):

        with open(self.visgen_config.region_phrase_txt, "r") as f:
            lines = f.readlines()
            sentences = [str.split(line) for line in lines]
            for size in [50, 100, 200, 300]:
                model_file_path = os.path.join(self.visgen_config.region_phrase_w2v_dir,
                                               "phrase_word2vec_" + str(size) + ".model")
                print("begin building {}".format(model_file_path))
                model = Word2Vec(sentences, size=size, window=5, min_count=1, workers=4)
                model.save(model_file_path)
                print("end building {}".format(model_file_path))
        pass


if __name__ == '__main__':
    w2v_utils = Word2VectorUtils()
    w2v_utils.build_w2v()
    pass
