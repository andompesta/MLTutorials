import argparse
from datetime import datetime
import vizdoom as vz
import random                # Handling random number generation
import time
from torchvision import transforms
import numpy as np
from RL.DuelingDoubleDQN import train
from RL.DuelingDoubleDQN.model_dd_dqn import DDDQN_Network
import torch
from visdom import Visdom
from RL.helper import frame_processor
from os import path
EXP_NAME = "exp-{}".format(datetime.now())

def __pars_args__():
    parser = argparse.ArgumentParser(description='DD-DQN')

    parser.add_argument('--max_grad_norm', type=float, default=100, help='value loss coefficient (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00025, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size used during learning')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./video', help='Path to save videos of agent')

    parser.add_argument("-u_target", "--update_target_estimator_every", default=350,
                        help="how ofter update the parameters of the target network")
    parser.add_argument("-ne", "--num_episodes", type=int, default=100000, help="Number of episodes to run for")
    # parser.add_argument('-a', '--actions', type=list, default=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], help='possible actions')
    parser.add_argument('-a', '--actions', type=list, default=[[1, 0], [0, 1]],
                        help='possible actions')
    parser.add_argument('-nf', '--number_frames', type=int, default=4, help='number of frame for each state')
    parser.add_argument('-ds', '--discount_factor', type=float, default=0.95, help='Reward discount factor')
    parser.add_argument("-es", "--epsilon_start", type=float, default=0.9, help="starting epsilon")
    parser.add_argument("-ee", "--epsilon_end", type=float, default=0.01, help="ending epsilon")
    parser.add_argument("-ed", "--epsilon_decay_rate", type=float, default=0.0005,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("-rv", "--record_video_every", type=int, default=50, help="Record a video every N episodes")
    parser.add_argument("-rm", "--replay_memory_size", type=int, default=10000, help="Size of the replay memory")
    parser.add_argument("-rm_init", "--replay_memory_init_size", type=int, default=500,
                        help="Number of random experiences to sample when initializing the reply memory")
    parser.add_argument("--max_steps", type=int, default=100, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[120, 120], help="Frame size")
    parser.add_argument("-uc", "--use_cuda", type=bool, default=False, help="Use cuda")
    parser.add_argument("-v", "--version", type=str, default="v-0", help="Use cuda")
    parser.add_argument("--show_window", type=bool, default=False, help="Show environment window")

    return parser.parse_args()

def create_enviroment(args):
    # game = vz.DoomGame()
    # game.load_config(path.join("../", "doom_setup", "doom_config.cfg"))
    # game.set_doom_scenario_path(path.join("../", "doom_setup", "basic.wad"))
    # game.set_window_visible(args.show_window)
    # game.init()
    import gym
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    return env

def test_environment(args):
    import matplotlib.pyplot as plt
    import itertools
    env = create_enviroment(args)

    crop = (0, 0, 0, 0)
    img_trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(args.state_size),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    episodes = 10
    for i in range(episodes):
        # env.new_episode()
        env.reset()

        for t in itertools.count():

            frame = env.render(mode='rgb_array')
            action_idx = env.action_space.sample()
            action = args.actions[action_idx]
            _, reward, done, _ = env.step(action_idx)
            print(action, reward)

            frame = frame_processor(frame, (100, 100,0,0), img_trans)

            plt.figure()
            plt.imshow(frame[:,:].numpy())
            plt.show()


            if done:
                break

        # while not done:
        #     state = game.get_state()
        #     frame = np.expand_dims(state.screen_buffer, axis=-1)
        #     misc = state.game_variables
        #     frame = frame_processor(frame, crop, img_trans)
        #
        #     plt.figure()
        #     plt.imshow(frame[:,:, 0].numpy())
        #     plt.show()
        #
        #
        #     print(action)
        #     reward = game.make_action(action)
        #     print("\treward:", reward)
        #     time.sleep(0.02)
        # print("Result:", game.get_total_reward())
        time.sleep(2)
    env.close()



if __name__ == '__main__':
    args = __pars_args__()
    # test_environment(args)
    env = create_enviroment(args)
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    print("record_video_every:{}\treplay_memory_size:{}\treplay_memory_init_size:{}".format(args.record_video_every,
                                                                                            args.replay_memory_size,
                                                                                            args.replay_memory_init_size))
    vis = Visdom()

    q_network = DDDQN_Network(args.batch_size, len(args.actions), args.number_frames,
                            kernels_size=[8, 4, 4],
                            out_channels=[32, 64, 128],
                            strides=[4, 2, 2],
                            fc_size=[3200, 1024])

    t_network = DDDQN_Network(args.batch_size, len(args.actions), args.number_frames,
                            kernels_size=[8, 4, 4],
                            out_channels=[32, 64, 128],
                            strides=[4, 2, 2],
                            fc_size=[3200, 1024])


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

    q_network.reset_parameters()
    t_network.reset_parameters()

    q_network.train()
    t_network.eval()

    for t, stats in train.work(env, q_network, t_network, args, vis, EXP_NAME,
                               optimizer, device):
        print("\nEpisode Reward: {}".format(stats.episode_reward))

