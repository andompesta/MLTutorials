import numpy as np
from torchvision import transforms
import os
import torch
import random
from collections import namedtuple


use_cuda = torch.cuda.is_available()

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class EpisodeStat(object):
    history = []
    def __init__(self, episode_lengths, episode_rewards):
        self.episode_length = episode_lengths
        self.episode_reward = episode_rewards
        self.history.append(episode_rewards)
        self.avg_reward = np.mean(self.history)


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




def stack_frame_setup(state_size, top_offset_height=30, bottom_offset_height=10, left_offset_width=30, right_offset_width=30):

    crop = (top_offset_height, bottom_offset_height, left_offset_width, right_offset_width)
    img_trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(state_size),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    def stack_frame(stacked_frames, state):
        """
        stack frame by frame on a queue of fixed length
        :param stacked_frames: 
        :param state: 
        :return: 
        """

        state = state_processor(state, crop, img_trans)
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(state)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.stack(stacked_frames, dim=0)
        return stacked_state
    return stack_frame

def state_processor(state, crop_size, transformations):
    """
    Processes a raw Atari iamges.
    Crop the image according to the offset passed and apply the define transitions.
    No needed to make it batched because we process one frame at a time, while the network is trained in  batch trough 
    experience replay
    :param state: A [210, 160, 3] Atari RGB State
    :param crop_size: quatruple containing the crop offsets
    :param transformations: image transformations to apply

    :return: A processed [84, 84, 1] state representing grayscale values.
    """
    state = img_crop_to_bounding_box(state, *crop_size)
    state = transformations(state)
    return state.permute(1,2,0)

def img_crop_to_bounding_box(img, top_offset_height, bottom_offset_height, left_offset_width, right_offset_width):
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
        h, w, c = image_shape
        img = img[top_offset_height:(h-bottom_offset_height), left_offset_width:(w-right_offset_width)]
        return img
    else:
        b, h, w, c = image_shape
        return img[:, top_offset_height:(h-bottom_offset_height), left_offset_width:(w-right_offset_width)]


def ensure_dir(file_path):
    '''
    Used to ensure to create the a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path
