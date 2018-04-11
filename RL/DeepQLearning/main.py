import argparse
from datetime import datetime
import vizdoom as vz
import random                # Handling random number generation
import time

from RL.DeepQLearning import train
from RL.DeepQLearning.model_dqn import DQN_Network
from torch import optim
from visdom import Visdom
from RL.DeepQLearning.helper import use_cuda
from os import path
EXP_NAME = "exp-{}".format(datetime.now())

def __pars_args__():
    parser = argparse.ArgumentParser(description='DQN')

    parser.add_argument('--max_grad_norm', type=float, default=100, help='value loss coefficient (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00025, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size used during learning')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./video', help='Path to save videos of agent')

    parser.add_argument("-u_target", "--update_target_estimator_every", default=2000,
                        help="how ofter update the parameters of the target network")
    parser.add_argument("-ne", "--num_episodes", type=int, default=100000, help="Number of episodes to run for")
    parser.add_argument('-a', '--actions', type=list, default=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], help='possible actions')
    parser.add_argument('-nf', '--number_frames', type=int, default=4, help='number of frame for each state')
    parser.add_argument('-ds', '--discount_factor', type=float, default=0.99, help='Reward discount factor')
    parser.add_argument("-es", "--epsilon_start", type=float, default=1., help="starting epsilon")
    parser.add_argument("-ee", "--epsilon_end", type=float, default=0.01, help="ending epsilon")
    parser.add_argument("-ed", "--epsilon_decay_rate", type=float, default=0.001,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("-rv", "--record_video_every", type=int, default=50, help="Record a video every N episodes")
    parser.add_argument("-rm", "--replay_memory_size", type=int, default=10000, help="Size of the replay memory")
    parser.add_argument("-rm_init", "--replay_memory_init_size", type=int, default=500,
                        help="Number of random experiences to sample when initializing the reply memory")
    parser.add_argument("--max_steps", type=int, default=100, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[84, 84], help="Max step for an episode")

    return parser.parse_args()

def create_enviroment():
    game = vz.DoomGame()
    game.load_config(path.join("doom_setup", "doom_config.cfg"))
    game.set_doom_scenario_path(path.join("doom_setup", "basic.wad"))
    game.init()
    return game

def test_environment(actions):
    game = vz.DoomGame()
    game.load_config(path.join("doom_setup", "doom_config.cfg"))
    game.set_doom_scenario_path(path.join("doom_setup", "basic.wad"))
    game.init()

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()



if __name__ == '__main__':
    args = __pars_args__()
    env = create_enviroment()

    print("record_video_every:{}\treplay_memory_size:{}\treplay_memory_init_size:{}".format(args.record_video_every,
                                                                                            args.replay_memory_size,
                                                                                            args.replay_memory_init_size))
    vis = Visdom()

    q_network = DQN_Network(args.batch_size, len(args.actions), args.number_frames,
                            kernels_size=[8, 4, 4],
                            out_channels=[32, 64, 128],
                            strides=[4, 2, 2],
                            fc_size=[1152, 512])

    t_network = DQN_Network(args.batch_size, len(args.actions), args.number_frames,
                            kernels_size=[8, 4, 4],
                            out_channels=[32, 64, 128],
                            strides=[4, 2, 2],
                            fc_size=[1152, 512])
    t_network.eval()

    if use_cuda:
        q_network.cuda()
        t_network.cuda()

    for t, stats in train.work(env, q_network, t_network, args, vis, EXP_NAME,
                               optim.Adagrad(q_network.parameters(), lr=args.learning_rate)):
        print("\nEpisode Reward: {}".format(stats.episode_rewards))

