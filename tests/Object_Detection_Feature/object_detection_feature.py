
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) before you start.

import os
import random
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../../")


# ## Object detection imports
# Here are the imports from the object detection module.


from util import label_map_util

from util import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
BASE_MODEL_PATH = '/home/zutnlp/data/ssd_mobilenet_v1_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = BASE_MODEL_PATH + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/zutnlp/data/label_mapping/mscoco_label_map.pbtxt'

#Path for Ai Challenge train data
PATH_TO_AI_CHALLENGE_TRAIN = '/home/zutnlp/data/challenger_train/ai_challenger_caption_train_20170902/caption_train_images_20170902'

NUM_CLASSES = 90




# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
def iterbrowse(path):
    for home, dirs, files in os.walk(path):
        for filename in files:
            yield os.path.join(home, filename)
def get_img_id(img_path):
    arrays = img_path.split('/')
    return arrays[len(arrays) -1].split('.')[0]
# gen  img feature for ai challenge train data :
TRAIN_IMAGE_PATHS = []
for full_path in iterbrowse(PATH_TO_AI_CHALLENGE_TRAIN):
    TRAIN_IMAGE_PATHS.append(full_path)
print('train data size: '+ str(len(TRAIN_IMAGE_PATHS)))
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    result_list = []
    for image_path in  TRAIN_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # gen img feature
      # img_id width heigth bottem_left upper_right
      img_id = get_img_id(image_path)
      print ('processing for img id:'+img_id)
      (img_width,img_height) = image.size
      # confidence score > 0.5,will save the feature
      confidence_score_list = np.squeeze(scores)
      classes_list = np.squeeze(classes).astype(np.int32)
      boxes_list = np.squeeze(boxes)
      for index in range(0,len(confidence_score_list)):
          cofidence_score = confidence_score_list[index]
          print (cofidence_score)
          if cofidence_score > 0.5:
              x_min= boxes_list[index][0] * img_width
              y_min= boxes_list[index][1] * img_height
              x_max= boxes_list[index][2] * img_width
              y_max= boxes_list[index][3] * img_height
              value = (img_id,
                     cofidence_score,
                     img_width,
                     img_height,
                     category_index.get(classes_list[index]).get('name'),
                     x_min,
                     y_min,
                     x_max,
                     y_max
                     )
              result_list.append(value)
    column_name = ['filename','confidence', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    label_df = pd.DataFrame(result_list, columns=column_name)
    label_df.to_csv('/home/zutnlp/data/label_mapping/ai_challenge_labels.csv', index=None)
    print('Successfully converted img feature to csv.')
