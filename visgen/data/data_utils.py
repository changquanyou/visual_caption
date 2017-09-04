import os

import skimage.color
import skimage.io
from skimage.transform import resize

from tf_visgen.visgen.config import VisgenDataConfig


def load_image(image_id):
    """
    load image data base on given image id
    :param image_id: image id
    :return: image data
    """
    current_image = skimage.io.imread(os.path.join(VisgenDataConfig.image_dir, str(image_id) + ".jpg"))
    if len(current_image.shape) == 2 or current_image.shape[2] == 1:  # this is to convert a gray to RGB image
        current_image = skimage.color.gray2rgb(current_image)  # 
    return current_image



def load_region(region, current_image):
    """
    get region from current image
    :param region: region data
    :param current_image: current image
    :return: region image data
    """
    region_image = current_image[region.y: region.y + region.height, region.x: region.x + region.width, :]
    # region_image = resize(region_image, (224, 224), mode='reflect')
    return region_image
