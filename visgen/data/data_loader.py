# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import json
import random

import ijson
import tensorflow as tf
from skimage.transform import resize

from tf_visgen.visgen.config import VisgenDataConfig
from tf_visgen.visgen.data import data_utils
from tf_visgen.visgen.models import Image, Object, Attribute, Relationship, Region


class DataLoader(object):
    def __init__(self, config):
        self.config = config

    def load_images(self):
        data_batch = []
        id_batch = []
        count = 1
        with open(file=self.config.image_data_json, mode='r', encoding='utf-8') as f:
            image_region_generator = ijson.items(f, "item")
            for data in image_region_generator:
                image_data = Image(image_id=data['image_id'], url=data['url'],
                                   width=data['width'], height=data['height'],
                                   coco_id=data['coco_id'], flickr_id=data['flickr_id'])
                image = data_utils.load_image(image_data.image_id)
                id_batch.append(image_data.image_id)

                image = resize(image, (224, 224), mode='reflect')
                data_batch.append(image)

                if count % self.config.batch_size == 0:
                    yield id_batch, data_batch
                    data_batch = []
                    id_batch = []
                count += 1
        yield id_batch, data_batch
        del id_batch, data_batch

    def load_regions(self):
        batch_data = []
        with open(file=self.config.region_descriptions_json, mode='r', encoding='utf-8') as f:
            image_region_generator = ijson.items(f, "item")
            for image_regions in image_region_generator:
                image_id = image_regions['id']
                regions = image_regions['regions']
                # current_image = data_utils.load_image(image_id)
                for r in regions:
                    region = Region(region_id=r['region_id'], image_id=r['image_id'],
                                    phrase=str.strip(r['phrase']), x=r['x'], y=r['y'],
                                    width=r['width'], height=r['height'])
                    batch_data.append(region)
                    if len(batch_data) == self.config.batch_size:
                        yield batch_data
                        batch_data = []
            if len(batch_data) > 0:
                yield batch_data
            del batch_data

    def sample_regions(self):
        """
        :return:
        """
        data_list = []
        count = 1
        with open(self.config.region_descriptions_json, mode='r', encoding='utf-8') as f:
            region_generator = ijson.items(f, "item")
            for image_regions in region_generator:
                regions = image_regions['regions']
                for r in regions:
                    random_num = random.uniform(0.0, 1.0)
                    if random_num <= 0.01:
                        data_list.append(r)
                    if count % 10000 == 0:
                        print("sampled data size is {} from {}".format(len(data_list), count))
                    count += 1
        with open(self.config.image_region_sample_txt, mode='a', encoding='utf-8') as f:
            json.dump(data_list, f)
        print("sampled data size is {} from {}".format(len(data_list), count))

    def load_objects(self):

        object_list = []

        with open(file=self.config.objects_json, mode='r', encoding='utf-8') as f:
            item_generator = ijson.items(f, "item")
            for data in item_generator:
                image_id = data["image_id"]
                objects = data["objects"]
                for obj_json in objects:
                    object_id = obj_json["object_id"]
                    h = obj_json["h"]
                    w = obj_json["w"]
                    x = obj_json["x"]
                    y = obj_json["y"]
                    names = obj_json["names"]
                    synsets = obj_json["synsets"]
                    obj = Object(object_id=object_id, x=x, y=y, height=h, width=w, names=names, synsets=synsets)
                    object_list.append(obj)
                    # print("object_id={}, x={}, y={}, h={}, w={}, names={}, synsets={}".format(object_id, x, y, h, w,
                    #                                                                           names, synsets))
                    if len(object_list) == self.config.batch_size:
                        yield object_list
                        object_list = []
            if len(object_list) > 0:
                yield object_list
            del object_list

    def load_attribute(self):
        batch_data = []
        with open(file=self.config.attributes_json, mode='r', encoding='utf-8') as f:
            item_generator = ijson.items(f, "item")
            for data in item_generator:
                image_id = data["image_id"]
                attributes = data["attributes"]
                for obj_json in attributes:
                    object_id = obj_json["object_id"]
                    subject = obj_json["subject"]
                    attribute = obj_json["attribute"]
                    synset = obj_json["synset"]
                    att = Attribute(object_id=object_id, subject=subject, attribute=attribute, synset=synset)
                    batch_data.append(att)
                    print("attribute={}".format(attribute))
                    if len(batch_data) == self.config.batch_size:
                        yield batch_data
                        batch_data = []
            if len(batch_data) > 0:
                yield batch_data
            del batch_data

    def load_relationship(self):
        batch_data = []
        with open(file=self.config.relationships_json, mode='r', encoding='utf-8') as f:
            item_generator = ijson.items(f, "item")
            for data in item_generator:
                image_id = data["image_id"]
                relationships = data["relationships"]
                for obj_json in relationships:
                    relationship_id = obj_json["relationship_id"]
                    predicate = obj_json["predicate"]
                    synsets = obj_json["synsets"]
                    subject = obj_json["subject"]
                    object = obj_json["object"]
                    relationship = Relationship(relationship_id=relationship_id,
                                                subject_id=subject["object_id"],
                                                object_id=object["object_id"],
                                                predicate=predicate,
                                                synset=synsets)
                    batch_data.append(relationship)
                    print("image_id={}, relationship={}".format(image_id, relationship))
                    if len(batch_data) == self.config.batch_size:
                        yield batch_data
                        batch_data = []
            if len(batch_data) > 0:
                yield batch_data
            del batch_data


def main(_):
    print("main function")
    visgen_config = VisgenDataConfig()
    visgen_loader = DataLoader(config=visgen_config)
    data_gen = visgen_loader.load_regions()
    for batch_order, batch_data in enumerate(data_gen):
        print("batch_order={}, size of batch_data={}".format(batch_order, len(batch_data)))

    pass


if __name__ == '__main__':
    tf.app.run()
