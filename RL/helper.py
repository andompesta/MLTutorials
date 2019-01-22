import numpy as np
from torchvision import transforms
import os
import torch
import random
from collections import namedtuple, deque
from PIL import Image

import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

img_trans = transforms.Compose([transforms.ToPILImage(),
                                    # transforms.Resize(state_size),
                                    transforms.Resize(40, interpolation=Image.CUBIC),
                                    transforms.ToTensor()])

def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    screen = frame_processor(screen, (0,0,0,0), img_trans)
    return screen

class EpisodeStat(object):
    history_rew = deque(maxlen=10)
    history_len = deque(maxlen=10)
    def __init__(self, episode_length, episode_reward):
        self.episode_length = episode_length
        self.episode_reward = episode_reward
        self.history_rew.append(episode_reward)
        self.history_len.append(episode_length)
        self.avg_reward = np.mean(self.history_rew)
        self.avg_length = np.mean(self.history_len)




class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return (random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


class ExperienceBuffer(object):
    def __init__(self, buffer_size=10000):
        '''
        store a history of experiences that can be randomly drawn from when training the network. We can draw form the
        previous past experiment to learn
        :param buffer_size: size of the buffer
        '''
        self.buffer = []
        self.buffer_size = buffer_size

    def store(self, experience):
        if len(list(self.buffer)) + len(list(experience)) >= self.buffer_size:
            self.buffer[0:(len(list(experience)) + len(list(self.buffer))) - self.buffer_size] = []
        self.buffer.extend([experience])

    def sample(self, size):
        samples = (random.sample(self.buffer, size))

        # state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*samples)
        #
        # state_batch = torch.stack(state_batch)
        # action_batch = torch.FloatTensor(list(action_batch))
        # reward_batch = torch.FloatTensor(reward_batch)
        # try:
        #     next_state_batch = torch.stack(next_state_batch).type(torch.FloatTensor)
        # except Exception as e:
        #     print(next_state_batch)
        #     raise e
        # done_batch = torch.FloatTensor(done_batch)

        return samples




def stack_frame_setup(state_size, top_offset_height=30, bottom_offset_height=10, left_offset_width=30, right_offset_width=30):

    crop = (top_offset_height, bottom_offset_height, left_offset_width, right_offset_width)


    img_trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(state_size),
                                    # transforms.Resize(40, interpolation=Image.CUBIC),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])


    def stack_frame(stacked_frames, frame, is_new_episode=False):
        """
        stack frame by frame on a queue of fixed length
        :param stacked_frames: deque used to mantain frame history
        :param frame: current frame
        :param is_new_episode: is a new episode
        :return: 
        """

        frame = frame_processor(frame, crop, img_trans)
        assert len(frame.shape) == 2

        if is_new_episode:
            stacked_frames.clear()
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)

        # Build the state (first dimension specifies different frames)
        state = torch.stack(list(stacked_frames), dim=0)
        return state
    return stack_frame

def frame_processor(frame, crop_size, transformations):
    """
    Processes a raw Atari iamges.
    Crop the image according to the offset passed and apply the define transitions.
    No needed to make it batched because we process one frame at a time, while the network is trained in  batch trough 
    experience replay
    :param frame: A [210, 160, 3] Atari RGB State
    :param crop_size: quatruple containing the crop offsets
    :param transformations: image transformations to apply

    :return: A processed [84, 84, 1] state representing grayscale values.
    """
    if len(frame.shape) == 2:
        # add channel dim
        frame = np.expand_dims(frame, axis=-1)

    frame = img_crop_to_bounding_box(frame, *crop_size)
    frame = transformations(frame)

    # plt.figure()
    # plt.imshow(frame[0].numpy(), cmap="gray")
    # plt.show()

    return frame

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
    if len(image_shape) == 2:
        h, w = image_shape
        img = img[top_offset_height:(h - bottom_offset_height), left_offset_width:(w - right_offset_width)]
        return img
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


def save_checkpoint(state, path, filename='checkpoint.pth.tar', version=0):
    torch.save(state, ensure_dir(os.path.join(path, version, filename)))


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.prob_alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def store(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0 ** self.prob_alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        total = len(self.buffer)

        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min * total) ** (-beta)

        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5) ** self.prob_alpha

    def __len__(self):
        return len(self.buffer)