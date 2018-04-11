import numpy as np
from skimage import transform
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
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*samples)

        state_batch = torch.stack(state_batch)
        action_batch = torch.FloatTensor(list(action_batch))
        reward_batch = torch.FloatTensor(reward_batch)
        try:
            next_state_batch = torch.stack(next_state_batch).type(torch.FloatTensor)
        except Exception as e:
            print(next_state_batch)
            raise e
        done_batch = torch.FloatTensor(done_batch)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch




def stack_frame_setup(state_size, offset_height=30, offset_width=30, target_height=80, target_width=130):
    def stack_frame(stacked_frames, state):
        """
        stack frame by frame on a queue of fixed length
        :param stacked_frames: 
        :param state: 
        :return: 
        """
        frame = state_processor(state, offset_height, offset_width, target_height, target_width, state_size)
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.stack(stacked_frames, dim=0)
        return stacked_state
    return stack_frame

def state_processor(state, offset_height, offset_width, target_height, target_width, state_size):
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    No needed to make it batched because we process one frame at a time, while the network is trained in  batch trough 
    experience replay
    :param state: A [210, 160, 3] Atari RGB State
    :return: A processed [84, 84, 1] state representing grayscale values.
    """
    img_size = state.shape

    # image = img_rgb2gray(state)      # convert to grayscale
    image = img_crop_to_bounding_box(state, offset_height, offset_width, target_height, target_width)
    image = image/255.
    image = img_resize(image, state_size)                # check aspect ration, otherwise convolution dim would not work
    return torch.from_numpy(image).type(torch.FloatTensor)

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

def img_resize(img, size):
    """
    resize a given image
    :param img: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor of shape `[height, width, channels]`.
    :param resize_factor: float or array for each axis
    :return:
    """
    return transform.resize(img, size)

def ensure_dir(file_path):
    '''
    Used to ensure to create the a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path
