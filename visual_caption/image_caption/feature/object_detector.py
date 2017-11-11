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
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from visual_caption.image_caption.data.data_config import ImageCaptionDataConfig

home = str(Path.home())  # home dir
base_data_dir = os.path.join(home, 'data')
model_data_dir = os.path.join(base_data_dir, "tf/models/object_detection")

MODEL_NAME = "faster_rcnn_nas_lowproposals_coco_2017_11_08"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(model_data_dir, MODEL_NAME + '/frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(model_data_dir, 'data' + '/mscoco_label_map.pbtxt')

NUM_CLASSES = 90
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


class DetectorConfig():
    model_ckpt = PATH_TO_CKPT
    path_labels = PATH_TO_LABELS
    num_classes = NUM_CLASSES
    image_size = IMAGE_SIZE


class ImageObjectDetector(object):
    def __init__(self, sess, detection_graph):
        self.config = DetectorConfig()
        self.sess = sess
        self.detection_graph = detection_graph
        self._build_inputs()
        self._load_label_map()

    def _build_inputs(self):
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.fetches = [detection_boxes, detection_scores, detection_classes, num_detections]

    @staticmethod
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def _load_label_map(self):
        PATH_TO_LABELS = self.config.path_labels
        NUM_CLASSES = self.config.num_classes
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def detect(self, image_path):
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        result = self.sess.run(fetches=self.fetches,
                               feed_dict={self.image_tensor: image_np_expanded})
        (boxes, scores, classes, num) = result
        return image_np, result

    def show(self, result):
        # Visualization of the results of a detection.

        image_np, (boxes, scores, classes, num) = result
        IMAGE_SIZE = self.config.image_size
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=1)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)


batch_size = 40


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


def main(_):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess_config = tf.ConfigProto(gpu_options=gpu_options,
                                 allow_soft_placement=True,
                                 log_device_placement=False)
    image_gen = load_images()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with tf.Session(graph=detection_graph, config=sess_config) as sess:
                detector = ImageObjectDetector(sess=sess,detection_graph=detection_graph)
                for batch, image_batch in enumerate(image_gen):
                    for idx, image_file in enumerate(image_batch):
                        result = detector.detect(image_path=image_file)
                        detector.show(result=result)


if __name__ == '__main__':
    tf.app.run()
