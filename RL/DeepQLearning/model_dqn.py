import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from RL.DeepQLearning.helper import NetworkType
import random

f_type = torch.cuda.FloatTensor
u_type = torch.cuda.ByteTensor
i_type = torch.cuda.LongTensor


class ExperienceBuffer():
    def __init__(self, buffer_size=50000):
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
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = tuple(map(np.array, zip(*samples)))
        return state_batch, action_batch, f_type(reward_batch), next_state_batch, f_type(done_batch)


def epsilon_greedy_policy(network):
    """
    Create a epsilon greedy policy function based on the given network Q-function
    :param network: network used to approximate the Q-function
    :return: action
    """
    def policy_fn(observation, epsilon):
        A = f_type(np.ones(network.action_space)) * epsilon / network.action_space
        q_values = network.forward(np.expand_dims(observation, axis=0))[0]
        best_action = torch.max(q_values, dim=0)[1]
        A[best_action] += (1.0 - epsilon)
        return A.cpu().numpy()
    return policy_fn

class DQN_Network(nn.Module):
    def __init__(self, batch_size, action_space, n_frames_input, kernels_size, out_channels, strides, fc_size, type_,
                 vis=None):
        """
        DQN netowrk
        """
        super(DQN_Network, self).__init__()

        self.batch_size = batch_size
        self.action_space = action_space
        self.n_frame_input = n_frames_input
        self.type = NetworkType(type_)
        if self.type._value_ == 1:
            self.step = 0

        assert len(out_channels) == 3
        self.out_channels = out_channels
        assert len(kernels_size) == 3
        self.kernels_size = kernels_size
        assert len(strides) == 3
        self.strides = strides

        assert len(fc_size) == 2
        self.fc_size = fc_size


        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.n_frame_input,
                                             out_channels=self.out_channels[0],
                                             kernel_size=(kernels_size[0], kernels_size[0]),
                                             stride=self.strides[0]),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.out_channels[0],
                                             out_channels=self.out_channels[1],
                                             kernel_size=(kernels_size[1], kernels_size[1]),
                                             stride=self.strides[1]),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=self.out_channels[1],
                                             out_channels=self.out_channels[2],
                                             kernel_size=(kernels_size[2], kernels_size[2]),
                                             stride=self.strides[2]),
                                   nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(in_features=self.fc_size[0],
                                           out_features=self.fc_size[1]),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=self.fc_size[1],
                                           out_features=self.action_space),
                                 nn.ReLU())

        self.mse_loss = nn.MSELoss()

    def forward(self, X):
        X = X/255.0
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = Variable(f_type(X))  # [batch_size, 80, 80, n_frames_input]

        X_conv1 = self.conv1(self.X_pl)
        X_conv2 = self.conv2(X_conv1)
        X_conv3 = self.conv3(X_conv2)

        X_flatten = X_conv3.view(X.shape[0], -1)
        X_fc1 = self.fc1(X_flatten)
        self.prediction = self.fc2(X_fc1)
        return self.prediction.data

    def compute_action_pred(self, actions):
        """
        Return the q_value for the given actions
        :param actions: a numpy array of action
        :return: 
        """
        # Integer id of which action was selected
        self.actions_pl = i_type(actions.tolist())  # [batch_size, 1]

        mask_index = i_type(range(self.batch_size)) * self.action_space + self.actions_pl
        self.action_predictions = self.prediction.view(-1)[mask_index]
        return self.action_predictions

    def compute_loss(self, target):
        """
        compute the loss between the q-values taken w.r.t the optimal ones
        :param target: target values obtained following an optimal policy
        :return: 
        """
        # The TD target value
        self.y_pl = Variable(f_type(target))  # [batch_size, 1]
        self.loss = self.mse_loss(self.action_predictions, self.y_pl)
        return self.loss
