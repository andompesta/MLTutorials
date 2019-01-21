import torch
from os import path
from RL.DQN.model_dqn import DQN_Network

model_path = path.join("RL", "DQN", "model")
q_network_chk = torch.load(path.join(model_path, "exp-2019-01-18 17:10:10.163325", "v-0", "q_net-679.cptk"))


actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
number_frames = 4
batch_size = 518
device = torch.device("cuda:0")

q_network = DQN_Network(batch_size, len(actions), number_frames,
                            kernels_size=[8, 4, 4],
                            out_channels=[32, 64, 128],
                            strides=[4, 2, 2],
                            fc_size=[3200, 512])
q_network.to(device)
q_network.load_state_dict(q_network_chk['state_dict'])
device = torch.device("cpu")
q_network.to(device)

torch.save(q_network.state_dict(), "q_net.cptk")