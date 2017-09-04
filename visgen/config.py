'''
@author: liuxiaoming
'''

import os

BATCH_SIZE = 40


class VisgenDataConfig(object):
    """
    Visual Genome Data sets Config Class
    """

    batch_size = BATCH_SIZE
    base_dir = "/home/liuxiaoming/data/"

    data_dir = os.path.join(base_dir, "visualgenome")
    image_dir = os.path.join(data_dir, "images")

    image_data_json = os.path.join(data_dir, "image_data.json")

    region_descriptions_json = os.path.join(data_dir, "region_descriptions.json")
    region_graphs_json = os.path.join(data_dir, "region_graphs.json")

    objects_json = os.path.join(data_dir, "objects.json")
    attributes_json = os.path.join(data_dir, "attributes.json")
    relationships_json = os.path.join(data_dir, "relationships.json")

    region_phrase_txt = os.path.join(data_dir, "region_phrase.txt")
    region_phrase_w2v_dir = os.path.join(data_dir, "word2vec")

    sample_dir = os.path.join(data_dir, "samples")
    image_region_sample_txt = os.path.join(sample_dir, "image_region_samples.txt")

    feature_dir = os.path.join(data_dir, "features")
    image_feature_file = os.path.join(feature_dir, "image_feature.mat")

    vgg_data_dir = os.path.join(base_dir, "vgg")

    word2vec_dir = os.path.join(data_dir, "word2vec")
    w2v_50_model_file = os.path.join(word2vec_dir, "w2v_50.model")

    google_vectors_file = os.path.join(word2vec_dir, "GoogleNews-vectors-negative300.bin")

    phrase_max_length = 64
