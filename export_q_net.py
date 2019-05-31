import torch
from os import path
from RL.DuelingDoubleDQN.model_dd_dqn import DDDQN_Network

model_path = path.join("model", "exp-dd-dqn", "v-deadly")
q_network_chk = torch.load(path.join(model_path, "q_net.cptk"))


actions = [0, 1, 2, 3, 4, 5, 6]
number_frames = 4
batch_size = 128
device = torch.device("cuda:0")

q_network = DDDQN_Network(batch_size, len(actions), number_frames,
                              kernels_size=[8, 4, 4],
                              out_channels=[32, 64, 64],
                              strides=[2, 2, 2],
                              fc_size=[4480, 512])
q_network.to(device)
q_network.load_state_dict(q_network_chk['state_dict'])
device = torch.device("cpu")
q_network.to(device)

torch.save(q_network.state_dict(), "q_net_exported.cptk")