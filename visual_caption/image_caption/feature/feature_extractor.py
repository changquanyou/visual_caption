# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from pathlib import Path

import skimage.color
import skimage.io
import tensorflow as tf
from PIL import Image
from scipy.misc import imresize
from sklearn.preprocessing import normalize

from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.utils.decorator_utils import timeit

slim = tf.contrib.slim
from slim.nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2

home = str(Path.home())  # home dir
base_data_dir = os.path.join(home, 'data')
model_data_dir = os.path.join(base_data_dir, "tf/models")
inception_resnet_v2_ckpt = os.path.join(model_data_dir, "inception_resnet_v2_2016_08_30.ckpt")

home = str(Path.home())  # home dir
base_data_dir = os.path.join(home, 'data')
model_data_dir = os.path.join(base_data_dir, "tf/models/object_detection")
MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08"
# What model to download.
BASE_MODEL_PATH = os.path.join(model_data_dir, MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = BASE_MODEL_PATH + '/frozen_inference_graph.pb'

batch_size = 40

import numpy as np


def load_image_raw(image_path):
    """
    load image data base on given image id
    :param image_path: image_path id
    :return: image data
    """
    try:
        current_image = skimage.io.imread(image_path)
        # this is to convert a gray to RGB image
        if len(current_image.shape) == 2 or current_image.shape[2] == 1:
            current_image = skimage.color.gray2rgb(current_image)  #
    except OSError as err:
        print(err)
        current_image = None
    return current_image


def load_image_into_list(image_files, image_shape):
    raw_images = list()
    for idx, image_path in enumerate(image_files):
        image_rawdata = load_image_raw(image_path=image_path)
        if image_rawdata is not None:
            image_rawdata = imresize(image_rawdata, image_shape)
            raw_images.append(image_rawdata)
    return raw_images


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


class FeatureExtractor(object):
    """
    inception_resnet_v2 feature extractor
    """

    def __init__(self, sess=None):
        self.input_images = tf.placeholder(shape=[None, 299, 299, 3],
                                           dtype=tf.float32, name='input_images')
        if sess is None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                         allow_soft_placement=True,
                                         log_device_placement=False)
            self.sess = tf.Session(config=sess_config)
        else:
            self.sess = sess
        self._load_model()

    @timeit
    def _load_model(self):
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), self.input_images)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_resnet_v2(scaled_input_tensor, is_training=False)
        saver = tf.train.Saver()
        checkpoint_file = inception_resnet_v2_ckpt
        saver.restore(self.sess, checkpoint_file)
        self.fetches = [end_points['PreLogitsFlatten'], logits]

    def get_features(self, images):
        raw_images = load_images(images)
        feed_dict = {self.input_images: raw_images}
        predict_values, logit_values = self.sess.run(self.fetches,
                                                     feed_dict)
        results = normalize(predict_values)
        return results


from object_detection.utils import label_map_util

NUM_CLASSES = 90
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(model_data_dir, 'data' + '/mscoco_label_map.pbtxt')


class FasterRCNN_FeatureExtractor(object):
    def __init__(self, config):
        self.config = config
        self._load_model()
        self._build_placehoulders()

    def _load_label_map(self):
        PATH_TO_LABELS = self.config.path_labels
        NUM_CLASSES = self.config.num_classes
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def _build_placehoulders(self):
        # Definite input and output Tensors for detection_graph
        self.imput_images = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # detection_features = self.detection_graph.get_tensor_by_name('Conv2d_7b_1x1:0')
        self.fetches = [detection_boxes, detection_scores, detection_classes, num_detections]

    @timeit
    def _load_model(self):
        checkpoint_path = self.config.model_ckpt
        summary_writer = tf.summary.FileWriter("./")

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                self.detection_graph = detection_graph
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                     allow_soft_placement=True,
                                     log_device_placement=False)
        self.sess = tf.Session(config=sess_config, graph=self.detection_graph)
        summary_writer.add_graph(self.sess.graph)
        self.summary_merged = tf.summary.merge_all()

        pass

    def detect(self, image_np_expanded):
        feed_dict = {self.imput_images: image_np_expanded}
        results = self.sess.run(fetches=self.fetches,
                                feed_dict=feed_dict)
        return results


def load_images():
    # data_config = VisualGenomeDataConfig()
    data_config = ImageCaptionDataConfig()
    image_files = list()
    for file_path in Path(data_config.train_image_dir).glob('**/*'):
        image_files.append(file_path.absolute())
        if len(image_files) == batch_size:
            yield image_files
            image_files = list()
    if len(image_files) > 0:
        yield image_files


class DetectorConfig():
    model_ckpt = PATH_TO_CKPT
    path_labels = PATH_TO_LABELS
    num_classes = NUM_CLASSES
    image_size = IMAGE_SIZE


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
                                                            max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def get_img_id(img_path):
    arrays = img_path.split('/')
    return arrays[len(arrays) - 1].split('.')[0]


def main(_):
    config = DetectorConfig()
    feature_extractor = FasterRCNN_FeatureExtractor(config=config)
    image_gen = load_images()
    for batch, batch_images in enumerate(image_gen):
        for idx, image_path in enumerate(batch_images):
            image = Image.open(image_path)
            (img_width, img_height) = image.size
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            result = feature_extractor.detect(image_np_expanded)

            boxes, scores, classes, num = result
            confidence_score_list = np.squeeze(scores)
            classes_list = np.squeeze(classes).astype(np.int32)
            boxes_list = np.squeeze(boxes)
            # img_id width heigth bottem_left upper_right
            img_id = get_img_id(str(image_path))
            print('processing for img id:' + img_id)
            for idx, confidence_score in enumerate(confidence_score_list):
                # feature = features[idx]
                box = boxes_list[idx]
                x_min = box[0] * img_width
                y_min = box[1] * img_height
                x_max = box[2] * img_width
                y_max = box[3] * img_height
                class_name = category_index.get(classes_list[idx]).get('name')
                if confidence_score > 0.5:
                    print("\tclass={}, score={} ".format(class_name, confidence_score))


if __name__ == '__main__':
    tf.app.run()
