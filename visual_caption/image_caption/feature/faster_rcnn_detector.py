# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig
from visual_caption.utils.decorator_utils import timeit

home = str(Path.home())  # home dir
base_data_dir = os.path.join(home, 'data')
model_data_dir = os.path.join(base_data_dir, "tf/models/object_detection")

MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08"
# What model to download.
BASE_MODEL_PATH = os.path.join(model_data_dir, MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = BASE_MODEL_PATH + '/frozen_inference_graph.pb'

NUM_CLASSES = 90
PATH_TO_LABELS = os.path.join(model_data_dir, 'data' + '/mscoco_label_map.pbtxt')


class DetectorConfig():
    model_ckpt = PATH_TO_CKPT
    path_labels = PATH_TO_LABELS
    num_classes = NUM_CLASSES


class FasterRCNNDetector(object):
    def __init__(self, config):
        self.config = config
        self._load_label_map()
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

    def detect_image(self, image_path):
        image = Image.open(image_path)
        (img_width, img_height) = image.size
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        detect_result = self.detect(image_np_expanded=image_np_expanded)
        boxes, scores, classes, num = detect_result
        confidence_score_list = np.squeeze(scores)
        classes_list = np.squeeze(classes).astype(np.int32)
        boxes_list = np.squeeze(boxes)

        results = list()
        for idx, confidence_score in enumerate(confidence_score_list):
            # feature = features[idx]
            box = boxes_list[idx]
            x_min = box[0] * img_width
            y_min = box[1] * img_height
            x_max = box[2] * img_width
            y_max = box[3] * img_height

            bbox_dict = {
                "x_min": x_min, "y_min": y_min,
                "x_max": x_max, "y_max": y_max
            }
            class_id = classes_list[idx]
            class_name = self.category_index.get(class_id).get('name')
            # if confidence_score > 0.5:
            data_dict = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence_score": confidence_score,
                "bbox": bbox_dict
            }
            results.append(data_dict)
            # print("\tidx={:4d}, class_name={:20}, score={:.8f}, real_box={}"
            #       .format(idx, class_name, confidence_score, bbox_dict))
        return results


def load_images():
    data_config = ImageCaptionDataConfig()
    batch_size = 40
    image_files = list()
    for file_path in Path(data_config.train_image_dir).glob('**/*'):
        image_files.append(file_path.absolute())
        if len(image_files) == batch_size:
            yield image_files
            image_files = list()
    if len(image_files) > 0:
        yield image_files


def get_img_id(img_path):
    arrays = img_path.split('/')
    return arrays[len(arrays) - 1].split('.')[0]


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def main(_):
    config = DetectorConfig()
    detector = FasterRCNNDetector(config=config)
    image_gen = load_images()
    for batch, batch_images in enumerate(image_gen):
        for idx, image_path in enumerate(batch_images):
            results = detector.detect_image(image_path=image_path)
            print("image_path={}".format(image_path))
            for idx, data_dict in enumerate(results):
                print("confidence_score={:.8f}, class_id={:2d}, class_name={:16}, bbox={}"
                      .format(data_dict["confidence_score"], data_dict["class_id"],
                              data_dict["class_name"], data_dict["bbox"]))


if __name__ == '__main__':
    tf.app.run()
