import argparse
from datetime import datetime
import vizdoom as vz
import random                # Handling random number generation
import time
from torchvision import transforms
import numpy as np
from RL.DQN import train
from RL.DQN.model_dqn import DQN_Network
from RL.DQN.model_dqn import epsilon_greedy_policy
import torch
from visdom import Visdom
from RL.DQN.helper import frame_processor
from os import path
import RL.helper as helper
from collections import deque

EXP_NAME = "exp-{}".format(datetime.now())

def __pars_args__():
    parser = argparse.ArgumentParser(description='DQN')

    parser.add_argument('--max_grad_norm', type=float, default=100, help='value loss coefficient (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00025, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=518, help='batch size used during learning')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./video', help='Path to save videos of agent')

    parser.add_argument("-u_target", "--update_target_estimator_every", default=500,
                        help="how ofter update the parameters of the target network")
    parser.add_argument("-ne", "--num_episodes", type=int, default=100000, help="Number of episodes to run for")
    parser.add_argument('-a', '--actions', type=list, default=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], help='possible actions')
    parser.add_argument('-nf', '--number_frames', type=int, default=4, help='number of frame for each state')
    parser.add_argument('-ds', '--discount_factor', type=float, default=0.99, help='Reward discount factor')
    parser.add_argument("-es", "--epsilon_start", type=float, default=0.1, help="starting epsilon")
    parser.add_argument("-ee", "--epsilon_end", type=float, default=0.01, help="ending epsilon")
    parser.add_argument("-ed", "--epsilon_decay_rate", type=float, default=0.0005,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("-rv", "--record_video_every", type=int, default=50, help="Record a video every N episodes")
    parser.add_argument("-rm", "--replay_memory_size", type=int, default=20000, help="Size of the replay memory")
    parser.add_argument("-rm_init", "--replay_memory_init_size", type=int, default=2500,
                        help="Number of random experiences to sample when initializing the reply memory")
    parser.add_argument("--max_steps", type=int, default=100, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[120, 120], help="Frame size")
    parser.add_argument("-uc", "--use_cuda", type=bool, default=False, help="Use cuda")
    parser.add_argument("-v", "--version", type=str, default="v-0", help="Use cuda")

    return parser.parse_args()

def create_enviroment():
    game = vz.DoomGame()
    game.load_config(path.join("doom_setup", "doom_config.cfg"))
    game.set_doom_scenario_path(path.join("doom_setup", "basic.wad"))
    game.init()
    return game



if __name__ == '__main__':
    args = __pars_args__()
    # test_environment(args)
    env = create_enviroment()
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    print("record_video_every:{}\treplay_memory_size:{}\treplay_memory_init_size:{}".format(args.record_video_every,
                                                                                            args.replay_memory_size,
                                                                                            args.replay_memory_init_size))

    # load previous weights
    q_network = torch.load(path.join(args.model_path, "q_net.cptk"))


    stack_frame_fn = helper.stack_frame_setup(args.state_size,
                                              top_offset_height=50,
                                              bottom_offset_height=10,
                                              left_offset_width=30,
                                              right_offset_width=30
                                              )

    for i in range(1):

        done = False

        env.new_episode()

        frame = env.get_state().screen_buffer
        stacked_frames = deque([torch.zeros(args.state_size) for i in range(args.number_frames)],
                               maxlen=args.number_frames)

        state = stack_frame_fn(stacked_frames, frame)

        while not env.is_episode_finished():
            b_state = state.unsqueeze(0).to("cpu")
            q_values = q_network.forward(b_state)[0]
            action_idx = torch.max(q_values, dim=0)[1]
            action = args.actions[action_idx.item()]

            env.make_action(action)
            done = env.is_episode_finished()
            score = env.get_total_reward()

            if done:
                break

            else:
                next_frame = env.get_state().screen_buffer
                next_state = stack_frame_fn(stacked_frames, next_frame)
                state = next_state

        score = env.get_total_reward()
        print("Score: ", score)
    env.close()

