import numpy as np
import torch
import torch.nn as nn
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
                input = observation.unsqueeze(0).to(device)
                q_values = network.forward(input)[0]
                best_action = torch.max(q_values, dim=0)[1]
                return best_action.cpu().item(), eps_threshold
        else:
            return np.random.randint(low=0, high=len(actions)), eps_threshold
    return policy_fn


# def epsilon_greedy_policy(network):
#     """
#     Create a epsilon greedy policy function based on the given network Q-function
#     :param network: network used to approximate the Q-function
#     :return: action
#     """
#     def policy_fn(observation, epsilon):
#         A = TENSOR_TYPE["f_tensor"](np.ones(network.action_space)) * epsilon / network.action_space
#         q_values = network.forward(Variable(observation.cuda(), volatile=True))[0]
#         best_action = torch.max(q_values, dim=0)[1]
#         A[best_action] += (1.0 - epsilon)
#         action = torch.multinomial(A, num_samples=1, replacement=True)
#         return action.cpu()
#     return policy_fn

class DQN_Network(nn.Module):
    def __init__(self, batch_size, action_space, n_frames_input, kernels_size, out_channels, strides, fc_size):
        """
        DQN netowrk
        """
        super(DQN_Network, self).__init__()

        self.batch_size = batch_size
        self.action_space = action_space
        self.n_frame_input = n_frames_input

        assert len(out_channels) == 3
        self.out_channels = out_channels
        assert len(kernels_size) == 3
        self.kernels_size = kernels_size
        assert len(strides) == 3
        self.strides = strides

        assert len(fc_size) == 2
        self.fc_size = fc_size

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                             out_channels=16,
                                             kernel_size=5,
                                             stride=2),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                             out_channels=32,
                                             kernel_size=5,
                                             stride=2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                             out_channels=32,
                                             kernel_size=5,
                                             stride=2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())

        # self.fc1 = nn.Sequential(nn.Linear(in_features=self.fc_size[0],
        #                                    out_features=self.fc_size[1]),
        #                          nn.ELU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=512,
                                           out_features=self.action_space))

        self._loss = nn.SmoothL1Loss()

    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, X):
        # Our input are 4 RGB frames of shape 160, 160 each
        batch_size = X.size(0)
        X_conv1 = self.conv1(X)
        X_conv2 = self.conv2(X_conv1)
        X_conv3 = self.conv3(X_conv2)

        X_flatten = X_conv3.view(batch_size, -1)
        # X_fc1 = self.fc1(X_flatten)
        prediction = self.fc2(X_flatten)
        return prediction

    def compute_q_value(self, states, actions):
        """
        Return the estimated_q_value for the given actions
        :param actions: an array of action
        :return: 
        """
        state_values = self.forward(states)
        # estimated_q_values = torch.sum(torch.mul(estimated_state_values, actions), dim=1, keepdim=True)
        state_action_values = state_values.gather(1, actions.unsqueeze(-1))
        return state_action_values

    def compute_loss(self, estimated_q_values, expected_q_values):
        """
        compute the loss between the q-values taken w.r.t the optimal ones
        :param estimated_q_values: current estimation of the state-action values obtained following an e-greedy policy
        :param expected_q_values: q_values obtained following an optimal policy on a previous network parametrization
        :return: 
        """
        # The TD target value

        self.loss = self._loss(estimated_q_values, expected_q_values)
        return self.loss
