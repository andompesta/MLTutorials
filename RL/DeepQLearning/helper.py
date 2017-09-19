import numpy as np
from scipy.ndimage import zoom
from enum import Enum
import os
import torch
import random

use_cuda = torch.cuda.is_available()
TENSOR_TYPE = dict(f_tensor=torch.cuda.FloatTensor if use_cuda else torch.FloatTensor,
                   i_tensor=torch.cuda.LongTensor if use_cuda else torch.LongTensor,
                   u_tensor=torch.cuda.ByteTensor if use_cuda else torch.ByteTensor)

class EpisodeStats(object):
    def __init__(self, episode_lengths, episode_rewards):
        self.episode_lengths = episode_lengths
        self.episode_rewards = episode_rewards

class ExperienceBuffer():
    def __init__(self, buffer_size=10000):
        '''
        store a history of experiences that can be randomly drawn from when training the network. We can draw form the
        previous past experiment to learn
        :param buffer_size: size of the buffer
        '''
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(list(self.buffer)) + len(list(experience)) >= self.buffer_size:
            self.buffer[0:(len(list(experience)) + len(list(self.buffer))) - self.buffer_size] = []
        self.buffer.extend([experience])

    def sample(self, size):
        samples = (random.sample(self.buffer, size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = tuple(zip(*samples))
        return torch.cat(state_batch).type(TENSOR_TYPE["f_tensor"]), torch.cat(action_batch).type(TENSOR_TYPE["i_tensor"]), torch.cat(TENSOR_TYPE["f_tensor"]([reward_batch])), \
               torch.cat(next_state_batch).type(TENSOR_TYPE["f_tensor"]), torch.cat(TENSOR_TYPE["f_tensor"]([done_batch]))




class NetworkType(Enum):
    TARGET = 1
    Q = 2

def img_rgb2gray(img):
    """
    convert rgb images to gray scale
    :param img: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
    :return:
    """
    image_shape = img.shape

    if len(image_shape) == 3:
        crop_img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        crop_img = np.dot(img[:, ..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
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
    if len(image_shape) == 2:
        return img[offset_height:offset_height + target_height, offset_width:target_width]
    else:
        return img[:, offset_height:offset_height + target_height, offset_width:target_width]

def img_resize(img, resize_factor, order=0):
    """
    resize a given image
    :param img: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
    :param resize_factor: float or array for each axis
    :return:
    """
    return zoom(img, zoom=resize_factor, order=order)


def state_processor(state, offset_height=34, offset_width=0, target_height=160, target_width=160):
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    No needed to make it batched because we process one frame at a time, while the network is trained in  batch trough 
    experience replay
    :param state: A [210, 160, 3] Atari RGB State
    :return: A processed [84, 84, 1] state representing grayscale values.
    """
    img_size = state.shape
    assert img_size[0] == 210
    assert img_size[1] == 160
    assert img_size[2] == 3


    image = img_rgb2gray(state)      # convert to rgb
    image = img_crop_to_bounding_box(image, 34, 0, 160, 160)
    image = img_resize(image, [0.525, 0.525])                # check aspect ration, otherwise convolution dim would not work
    return torch.from_numpy(image/255.).type(torch.FloatTensor).unsqueeze(dim=0)

def ensure_dir(file_path):
    '''
    Used to ensure to create the a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path
