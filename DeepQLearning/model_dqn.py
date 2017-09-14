import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


f_type = torch.FloatTensor
u_type = torch.ByteTensor
i_type = torch.IntTensor


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


        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.n_frame_input,
                                             out_channels=self.out_channels[0],
                                             kernel_size=kernels_size[0],
                                             stride=self.strides[0]),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.out_channels[0],
                                             out_channels=self.out_channels[1],
                                             kernel_size=kernels_size[1],
                                             stride=self.strides[1]),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=self.out_channels[1],
                                             out_channels=self.out_channels[2],
                                             kernel_size=kernels_size[2],
                                             stride=self.strides[2]),
                                   nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(in_features=self.fc_size[0],
                                           out_features=self.fc_size[1]),
                                 nn.ReLU())
        self.fc2 = nn.Linear(in_features=self.fc_size[1], out_features=self.action_space)


    def forward(self, X, y, actions):
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = Variable(u_type(X))  # [batch_size, 80, 80, n_frames_input]
        # The TD target value
        self.y_pl = Variable(u_type(y))  # [batch_size, 1]
        # Integer id of which action was selected
        self.actions_pl = Variable(i_type(actions))  # [batch_size, 1]