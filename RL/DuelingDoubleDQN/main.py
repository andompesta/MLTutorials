import argparse
from datetime import datetime
import vizdoom as vz
from  RL.envs.maze_env_gif import Maze
import random                # Handling random number generation
import time
from torchvision import transforms
import numpy as np
from RL.DuelingDoubleDQN.train import work
from RL.DuelingDoubleDQN.model_dd_dqn import DDDQN_Network
import torch
from visdom import Visdom
from RL.helper import frame_processor
from os import path
from RL.viz_wrapper import VizdoomEnv
EXP_NAME = "exp-dd-dqn"

def __pars_args__():
    parser = argparse.ArgumentParser(description='DD-DQN')

    parser.add_argument('--max_grad', type=float, default=30, help='value loss coefficient (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00025, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size used during learning')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./video', help='Path to save videos of agent')

    parser.add_argument("-u_target", "--update_target_estimator_every", default=20,
                        help="how ofter update the parameters of the target network")
    parser.add_argument("-ne", "--num_episodes", type=int, default=5000, help="Number of episodes to run for")
    parser.add_argument('-a', '--actions', type=list, default=[0, 1, 2, 3, 4, 5, 6],
                        help='possible actions')

    parser.add_argument('-nf', '--number_frames', type=int, default=4, help='number of frame for each state')
    parser.add_argument('-ds', '--discount_factor', type=float, default=0.95, help='Reward discount factor')
    parser.add_argument("-es", "--epsilon_start", type=float, default=1.0, help="starting epsilon")
    parser.add_argument("-ee", "--epsilon_end", type=float, default=0.01, help="ending epsilon")
    parser.add_argument("-ed", "--epsilon_decay_rate", type=float, default=0.0005,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("-rm", "--replay_memory_size", type=int, default=1000000, help="Size of the replay memory")
    parser.add_argument("-rm_init", "--replay_memory_init_size", type=int, default=100000,
                        help="Number of random experiences to sample when initializing the reply memory")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[80, 100], help="Frame size")
    parser.add_argument("-uc", "--use_cuda", type=bool, default=False, help="Use cuda")
    parser.add_argument("-v", "--version", type=str, default="v-doom", help="Model version")
    parser.add_argument("--show_window", type=bool, default=True, help="Show env windows")

    return parser.parse_args()

def test_environment(args):
    import matplotlib.pyplot as plt
    import itertools
    env = VizdoomEnv(path.join("..", "doom_setup", "deadly_corridor"), args.show_window, len(args.actions))

    img_trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.CenterCrop([170, 280]),
                                    transforms.Resize(args.state_size),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    action_map = np.identity(len(args.actions), dtype=int).tolist()
    # img_trans = transforms.Compose([transforms.Resize(args.state_size),
    #                                 transforms.Grayscale(),
    #                                 transforms.ToTensor()])

    episodes = 10
    for i in range(episodes):
        frame = env.reset()

        for t in itertools.count():
            frame = frame_processor(frame, img_trans)

            plt.figure()
            plt.imshow(frame[0, :, :].numpy(), cmap="gray")
            plt.show()

            action = random.choice(args.actions)
            frame, reward, done, info = env.step(action_map[action])

            print("\treward:", reward)
            time.sleep(0.02)

            if done:
                break

        time.sleep(2)
    env.close()


def run(args, env, device):
    vis = Visdom()

    q_network = DDDQN_Network(args.batch_size, len(args.actions), args.number_frames,
                              kernels_size=[8, 4, 4],
                              out_channels=[32, 64, 64],
                              strides=[2, 2, 2],
                              # fc_size=[3584, 512])
                              fc_size=[4480, 512])


    t_network = DDDQN_Network(args.batch_size, len(args.actions), args.number_frames,
                              kernels_size=[8, 4, 4],
                              out_channels=[32, 64, 64],
                              strides=[2, 2, 2],
                              fc_size=[4480, 512])
                              # fc_size=[3584, 512])



    q_network.reset_parameters()
    t_network.load_state_dict(q_network.state_dict())
    t_network.eval()

    q_network.to(device)
    t_network.to(device)

    optimizer = torch.optim.RMSprop(q_network.parameters(), lr=args.learning_rate)

    # load previous weights
    # q_network_chk = torch.load(path.join(args.model_path, "exp-2019-01-18 17:10:10.163325", "q_net-679.cptk"))
    # q_network.load_state_dict(q_network_chk['state_dict'])
    # optimizer.load_state_dict(q_network_chk['optimizer'])
    #
    # t_network_chk = torch.load(path.join(args.model_path, "exp-2019-01-18 17:10:10.163325", "t_net-679.cptk"))
    # t_network.load_state_dict(t_network_chk["state_dict"])

    for t, stats in work(env, q_network, t_network, args, vis, EXP_NAME, optimizer, device):
        print("\nEpisode Reward: {}".format(stats.episode_reward))

if __name__ == '__main__':
    args = __pars_args__()
    # test_environment(args)
    env = VizdoomEnv(path.join("..", "doom_setup", "deadly_corridor"), args.show_window, len(args.actions))
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    run(args, env, device)