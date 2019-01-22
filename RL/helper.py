import numpy as np
from torchvision import transforms
import os
import torch
import random
from collections import namedtuple, deque

import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

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
    return screen

class EpisodeStat(object):
    history_rew = []
    history_len = []
    def __init__(self, episode_length, episode_reward):
        self.episode_length = episode_length
        self.episode_reward = episode_reward
        self.history_rew.append(episode_reward)
        self.history_len.append(episode_length)
        self.avg_reward = np.mean(self.history_rew)
        self.avg_length = np.mean(self.history_len)



class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences)

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        """tree:
                    0
                   / \
                  0   0
                 / \ / \
        tree_index  0 0  0  We fill the leaves from left to right

        :param priority: priority of the update
        :param data: update
        :return:
        """
        tree_index = self.data_pointer + self.capacity - 1



        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0


    def update(self, tree_index, priority):
        """
        Update the leaf priority score and propagate the change through tree
        :param tree_index: index in the tree
        :param priority: priority to add
        :return:
        """
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """

            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        :param v:
        :return:
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            state, action, reward, next_state, done = data
            memory_b.append([state,
                             torch.tensor(action),
                             torch.tensor(reward),
                             next_state,
                             torch.tensor(done)
                             ])

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors.data.cpu().numpy(), self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)







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

    return frame[0]

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