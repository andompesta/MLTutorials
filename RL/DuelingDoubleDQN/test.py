import argparse
from datetime import datetime
import numpy as np
from torchvision import transforms as T
from RL.viz_wrapper import VizdoomEnv
from RL.DuelingDoubleDQN.model_dd_dqn import epsilon_greedy_policy, DDDQN_Network
import torch
from os import path
import RL.helper as helper
from collections import deque

EXP_NAME = "exp-{}".format(datetime.now())

def __pars_args__():
    parser = argparse.ArgumentParser(description='DD-DQN')

    parser.add_argument('-a', '--actions', type=list, default=[0, 1, 2, 3, 4, 5, 6],
                        help='possible actions')

    parser.add_argument('-nf', '--number_frames', type=int, default=4, help='number of frame for each state')
    parser.add_argument('-ds', '--discount_factor', type=float, default=0.95, help='Reward discount factor')
    parser.add_argument("-es", "--epsilon_start", type=float, default=0.01, help="starting epsilon")
    parser.add_argument("-ee", "--epsilon_end", type=float, default=0.01, help="ending epsilon")
    parser.add_argument("-ed", "--epsilon_decay_rate", type=float, default=0.0005,
                        help="Number of steps to decay epsilon over")

    parser.add_argument("--state_size", type=list, default=[80, 100], help="Frame size")

    return parser.parse_args()

if __name__ == '__main__':
    args = __pars_args__()
    env = VizdoomEnv(path.join("..", "doom_setup", "deadly_corridor"), True, len(args.actions))
    device = torch.device("cpu")

    IMG_TRANS = T.Compose([T.ToPILImage(),
                           T.CenterCrop([170, 280]),
                           T.Resize(args.state_size),
                           T.Grayscale(),
                           T.ToTensor()])

    ACTION_MAP = np.identity(len(args.actions), dtype=int).tolist()

    q_network = DDDQN_Network(1, len(args.actions), args.number_frames,
                              kernels_size=[8, 4, 4],
                              out_channels=[32, 64, 64],
                              strides=[2, 2, 2],
                              fc_size=[4480, 512])


    # load previous weights
    q_network.load_state_dict(torch.load(path.join("./model", "exp-dd-dqn", "v-deadly", "q_net_exported.cptk")))

    stack_frame_fn = helper.stack_frame_setup(IMG_TRANS)
    frames_queue = deque(maxlen=args.number_frames)


    for i in range(5):

        done = False

        frame = env.reset()
        state = stack_frame_fn(frames_queue, frame, True)
        tot_reward = 0
        framecount = 0
        while not env.env.is_player_dead() or env.env.is_episode_finished():
            q_values = q_network.forward(state)[0]
            action = torch.max(q_values, dim=0)[1]

            frame, reward, done, info = env.step(ACTION_MAP[action.item()])

            if framecount % 2 == True:
                print("stop")

            if done:
                break

            else:
                next_state = stack_frame_fn(frames_queue, frame)
                state = next_state

            framecount += 1
            tot_reward += reward

        print("Score: ", tot_reward)
    env.close()

