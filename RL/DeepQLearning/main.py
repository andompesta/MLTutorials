import argparse
from datetime import datetime
import gym
from RL.DeepQLearning import train
from RL.DeepQLearning.model_dqn import DQN_Network
from torch import optim
from visdom import Visdom

EXP_NAME = "exp-{}".format(datetime.now())

def __pars_args__():
    parser = argparse.ArgumentParser(description='DQN')

    parser.add_argument('--max_grad_norm', type=float, default=100, help='value loss coefficient (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00025, help='learning rate (default: 0.001)')
    parser.add_argument('--cuda', default=True, help='Execute on cuda')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size used during learning')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./video', help='Path to save videos of agent')

    parser.add_argument("-u_target", "--update_target_estimator_every", default=10000,
                        help="how ofter update the parameters of the target network")
    parser.add_argument("-ne", "--num_episodes", type=int, default=10000, help="Number of episodes to run for")
    parser.add_argument('-a', '--actions', type=list, default=[0, 1, 2, 3], help='possible actions')
    parser.add_argument('-nf', '--number_frames', type=int, default=4, help='number of frame for each state')
    parser.add_argument('-ds', '--discount_factor', type=float, default=0.99, help='Reward discount factor')
    parser.add_argument("-es", "--epsilon_start", type=float, default=1., help="starting epsilon")
    parser.add_argument("-ee", "--epsilon_end", type=float, default=0.1, help="ending epsilon")
    parser.add_argument("-ed", "--epsilon_decay_steps", type=int, default=500000,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("-rv", "--record_video_every", type=int, default=50, help="Record a video every N episodes")
    parser.add_argument("-rm", "--replay_memory_size", type=int, default=500000, help="Size of the replay memory")
    parser.add_argument("-rm_init", "--replay_memory_init_size", type=int, default=50,
                        help="Number of random experiences to sample when initializing the reply memory")
    return parser.parse_args()

if __name__ == '__main__':
    args = __pars_args__()
    vis = Visdom()
    env = gym.envs.make("Breakout-v0").unwrapped

    q_network = DQN_Network(args.batch_size, len(args.actions), args.number_frames,
                            kernels_size=[8, 4, 3],
                            out_channels=[32, 64, 64],
                            strides=[4, 2, 1],
                            fc_size=[3136, 512],
                            type_=2)

    t_network = DQN_Network(args.batch_size, len(args.actions), args.number_frames,
                            kernels_size=[8, 4, 3],
                            out_channels=[32, 64, 64],
                            strides=[4, 2, 1],
                            fc_size=[3136, 512],
                            type_=1)
    if args.cuda:
        q_network.cuda()
        t_network.cuda()

    for t, stats in train.work(env, q_network, t_network, args, vis, EXP_NAME,
                               optim.RMSprop(q_network.parameters(), lr=args.learning_rate)):
        print("\nEpisode Reward: {}".format(stats.episode_rewards))

