import numpy as np
import torch
import torch.nn as nn
import math
import random

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

                values = network.forward(observation.to(device))[0]
                best_action = torch.max(values, dim=0)[1]
                return best_action.cpu().item(), eps_threshold
        else:
            # return torch.tensor(np.random.randint(low=0, high=num_actions), dtype=torch.long), eps_threshold
            return random.choice(actions), eps_threshold
    return policy_fn

class DDDQN_Network(nn.Module):
    def __init__(self, batch_size, action_space, n_frames_input, kernels_size, out_channels, strides, fc_size):
        """
        DQN netowrk
        """
        super(DDDQN_Network, self).__init__()

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


        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.n_frame_input,
                                             out_channels=self.out_channels[0],
                                             kernel_size=kernels_size[0],
                                             stride=self.strides[0]),
                                   nn.BatchNorm2d(self.out_channels[0]),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.out_channels[0],
                                             out_channels=self.out_channels[1],
                                             kernel_size=kernels_size[1],
                                             stride=self.strides[1]),
                                   nn.BatchNorm2d(self.out_channels[1]),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=self.out_channels[1],
                                             out_channels=self.out_channels[2],
                                             kernel_size=kernels_size[2],
                                             stride=self.strides[2]),
                                   nn.BatchNorm2d(self.out_channels[2]),
                                   nn.ReLU())

        self.value_fc = nn.Sequential(nn.Linear(in_features=self.fc_size[0],
                                                out_features=self.fc_size[1]),
                                      nn.ReLU())
        self.value = nn.Linear(in_features=self.fc_size[1],
                               out_features=1)                              # value function for given state s_t

        self.advantage_fc = nn.Sequential(nn.Linear(in_features=self.fc_size[0],
                                                    out_features=self.fc_size[1]),
                                          nn.ReLU())
        self.advantage = nn.Linear(in_features=self.fc_size[1],
                                   out_features=self.action_space)          # advantage of each action at state s_t

        self._loss = nn.SmoothL1Loss(reduction='none')

    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, X):
        """
        Estimate the value function (expected future reward for the given input and every action)
        :param X: input state
        :return:
        """
        # Our input are 4 RGB frames of shape 160, 160 each
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)

        X_flatten = X.view(X.size(0), -1)
        value = self.value_fc(X_flatten)
        value = self.value(value)

        advantage = self.advantage_fc(X_flatten)
        advantage = self.advantage(advantage)

        state_value = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))

        return state_value

    def compute_q_value(self, state, actions):
        """
        Return the q_value for the given state and action
        :param state: state s_t obtained from the env
        :param actions: an array of action
        :return: 
        """
        state_values = self.forward(state)
        state_action_value = state_values.gather(1, actions)
        return state_action_value.squeeze()

    def compute_loss(self, state_action_values, next_state_values, weights):
        """
        compute the loss between the q-values taken w.r.t the optimal ones
        :param q_values: current estimation of the state-action values obtained following an e-greedy policy
        :param q_target: q_values obtained following an optimal policy on a previous network parametrization
        :param IS_weights: weight used to remove the bias of the priority sampling
        :return: 
        """

        absolute_error = torch.abs(next_state_values - state_action_values)
        td_error = self._loss(next_state_values, state_action_values)
        loss = torch.mean(weights * td_error)
        return loss, td_error.mean(), absolute_error
