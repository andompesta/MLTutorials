import numpy as np
from scipy.ndimage import zoom

def img_rgb2gray(img):
    """
    convert rgb images to gray scale
    :param img: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
    :return:
    """
    image_shape = img.shape

    if len(image_shape) == 3:
        crop_img = np.expand_dims(np.dot(img[..., :3], [0.299, 0.587, 0.114]), axis=-1).astype(np.uint8)
    else:
        crop_img = np.expand_dims(np.dot(img[:, ..., :3], [0.299, 0.587, 0.114]), axis=-1).astype(np.uint8)

    return crop_img


def img_crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width):
    """
    :param img:4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
    :param offset_height: Vertical coordinate of the top-left corner of the result in the input.
    :param offset_width: Horizontal coordinate of the top-left corner of the result in the input.
    :param target_height: Height of the result.
    :param target_width:Width of the result.
    :return:
    """
    image_shape = img.shape
    if len(image_shape) == 3:
        return img[offset_height:offset_height + target_height, offset_width:target_width, :]
    else:
        return img[:, offset_height:offset_height + target_height, offset_width:target_width, :]

def img_resize(img, resize_factor, order=0):
    """
    resize a given image
    :param img: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
    :param resize_factor: float or array for each axis
    :return:
    """
    return zoom(img, zoom=resize_factor, order=order)


def state_processor(state):
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    :param state: A [210, 160, 3] Atari RGB State
    :return: A processed [84, 84, 1] state representing grayscale values.
    """
    img_size = state.shape()
    assert img_size[0] == 210
    assert img_size[1] == 160
    assert img_size[2] == 3

    image = img_rgb2gray(state)      # convert to rgb
    image = img_crop_to_bounding_box(image, 34, 0, 160, 160)
    image = img_resize(image, [0.5, 0.5, 1])

    return image