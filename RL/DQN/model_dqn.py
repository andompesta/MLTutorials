import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


def epsilon_greedy_policy(network, eps_end, eps_start, eps_decay, actions, device):
    """
    Create a epsilon greedy policy function based on the given network Q-function
    :param network: network used to approximate the Q-function
    :return: action
    """
    def policy_fn(observation, steps_done):
        sample = np.random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done * eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                if observation.dim() == 3:
                    observation = observation.unsqueeze(0)
                elif observation.dim() < 3:
                    NotImplementedError("Wrong input dim")

                q_values = network.forward(observation)[0]
                best_action = torch.max(q_values, dim=0)[1]
                return best_action.cpu().item(), eps_threshold
        else:
            return np.random.randint(low=0, high=len(actions)), eps_threshold
    return policy_fn


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        self.head = nn.Linear(448, 4) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p)

# class DQN_Network(nn.Module):
#     def __init__(self, batch_size, action_space, n_frames_input, kernels_size, out_channels, strides, fc_size):
#         """
#         DQN netowrk
#         """
#         super(DQN_Network, self).__init__()
#
#         self.batch_size = batch_size
#         self.action_space = action_space
#         self.n_frame_input = n_frames_input
#
#         assert len(out_channels) == 3
#         self.out_channels = out_channels
#         assert len(kernels_size) == 3
#         self.kernels_size = kernels_size
#         assert len(strides) == 3
#         self.strides = strides
#
#         assert len(fc_size) == 2
#         self.fc_size = fc_size
#
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
#                                              out_channels=16,
#                                              kernel_size=5,
#                                              stride=2),
#                                    nn.BatchNorm2d(16),
#                                    nn.ReLU())
#
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,
#                                              out_channels=32,
#                                              kernel_size=5,
#                                              stride=2),
#                                    nn.BatchNorm2d(32),
#                                    nn.ReLU())
#
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,
#                                              out_channels=32,
#                                              kernel_size=5,
#                                              stride=2),
#                                    nn.BatchNorm2d(32),
#                                    nn.ReLU())
#
#         # self.fc1 = nn.Sequential(nn.Linear(in_features=self.fc_size[0],
#         #                                    out_features=self.fc_size[1]),
#         #                          nn.ELU())
#         self.fc2 = nn.Linear(in_features=512, out_features=self.action_space)
#
#     def reset_parameters(self):
#         for p in self.parameters():
#             if len(p.data.shape) > 1:
#                 nn.init.xavier_uniform_(p)
#
#
#     def forward(self, X):
#         # Our input are 4 RGB frames of shape 160, 160 each
#         batch_size = X.size(0)
#         X = self.conv1(X)
#         X = self.conv2(X)
#         X = self.conv3(X)
#
#         X = X.view(batch_size, -1)
#         # X_fc1 = self.fc1(X_flatten)
#         return self.fc2(X)
#
#     def compute_q_value(self, states, actions):
#         """
#         Return the estimated_q_value for the given actions
#         :param actions: an array of action
#         :return:
#         """
#         state_values = self.forward(states)
#         # estimated_q_values = torch.sum(torch.mul(estimated_state_values, actions), dim=1, keepdim=True)
#         state_action_values = state_values.gather(1, actions.unsqueeze(-1))
#         return state_action_values
#
#     def compute_loss(self, estimated_q_values, expected_q_values):
#         """
#         compute the loss between the q-values taken w.r.t the optimal ones
#         :param estimated_q_values: current estimation of the state-action values obtained following an e-greedy policy
#         :param expected_q_values: q_values obtained following an optimal policy on a previous network parametrization
#         :return:
#         """
#         # The TD target value
#
#         self.loss = self._loss(estimated_q_values, expected_q_values)
#         return self.loss
