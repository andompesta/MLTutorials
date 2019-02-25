import argparse
from datetime import datetime
import vizdoom as vz
from torchvision import transforms as T
import numpy as np
from RL.DuelingDoubleDQN.model_dd_dqn import epsilon_greedy_policy, DDDQN_Network
import torch
from os import path
import RL.helper as helper
from collections import deque

EXP_NAME = "exp-{}".format(datetime.now())

def __pars_args__():
    parser = argparse.ArgumentParser(description='DQN')

    parser.add_argument('--max_grad', type=float, default=30, help='value loss coefficient (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size used during learning')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./video', help='Path to save videos of agent')

    parser.add_argument("-u_target", "--update_target_estimator_every", default=20,
                        help="how ofter update the parameters of the target network")
    parser.add_argument("-ne", "--num_episodes", type=int, default=1000, help="Number of episodes to run for")
    # parser.add_argument('-a', '--actions', type=list, default=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], help='possible actions')
    parser.add_argument('-a', '--actions', type=list, default=[0, 1, 2],
                        help='possible actions')
    # parser.add_argument('-a', '--actions', type=int, default=2, help='possible actions')

    parser.add_argument('-nf', '--number_frames', type=int, default=4, help='number of frame for each state')
    parser.add_argument('-ds', '--discount_factor', type=float, default=0.95, help='Reward discount factor')
    parser.add_argument("-es", "--epsilon_start", type=float, default=1.0, help="starting epsilon")
    parser.add_argument("-ee", "--epsilon_end", type=float, default=0.01, help="ending epsilon")
    parser.add_argument("-ed", "--epsilon_decay_rate", type=float, default=0.001,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("-rv", "--record_video_every", type=int, default=50, help="Record a video every N episodes")
    parser.add_argument("-rm", "--replay_memory_size", type=int, default=1000000, help="Size of the replay memory")
    parser.add_argument("-rm_init", "--replay_memory_init_size", type=int, default=10000,
                        help="Number of random experiences to sample when initializing the reply memory")
    parser.add_argument("--max_steps", type=int, default=100, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[84, 84], help="Frame size")
    parser.add_argument("-uc", "--use_cuda", type=bool, default=False, help="Use cuda")
    parser.add_argument("-v", "--version", type=str, default="v-doom", help="Use cuda")
    parser.add_argument("--show_window", type=bool, default=True, help="Show env windows")

    return parser.parse_args()

def create_enviroment():
    env = vz.DoomGame()
    env.load_config(path.join("..", "doom_setup", "doom_basic.cfg"))
    env.set_doom_scenario_path(path.join("..", "doom_setup", "basic.wad"))
    env.set_window_visible(args.show_window)
    env.init()
    return env



if __name__ == '__main__':
    args = __pars_args__()
    env = create_enviroment()
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    IMG_TRANS = T.Compose([T.ToPILImage(),
                           T.CenterCrop([170, 200]),
                           T.Resize(args.state_size),
                           T.Grayscale(),
                           T.ToTensor()])

    ACTION_MAP = {
        0: [1, 0, 0],
        1: [0, 1, 0],
        2: [0, 0, 1]
    }

    q_network = DDDQN_Network(args.batch_size, len(args.actions), args.number_frames,
                              kernels_size=[8, 4, 4],
                              out_channels=[32, 64, 64],
                              strides=[2, 2, 2],
                              # fc_size=[3584, 512])
                              fc_size=[4096, 512])


    # load previous weights
    q_network.load_state_dict(torch.load(path.join(args.model_path, "exp-dd-dqn", "v-doom", "q_net.cptk")))

    stack_frame_fn = helper.stack_frame_setup(IMG_TRANS)



    # The policy we're following
    policy = epsilon_greedy_policy(q_network,
                                   args.epsilon_end,
                                   args.epsilon_start,
                                   args.epsilon_decay_rate,
                                   args.actions,
                                   device)
    frames_queue = deque(maxlen=args.number_frames)


    for i in range(1):

        done = False

        env.new_episode()
        frame = env.get_state().screen_buffer


        state = stack_frame_fn(frames_queue, frame, True)

        while not env.is_episode_finished():
            q_values = q_network.forward(state)[0]
            action = torch.max(q_values, dim=0)[1]


            env.make_action(ACTION_MAP[action.item()])
            done = env.is_episode_finished()
            score = env.get_total_reward()

            if done:
                break

            else:
                next_frame = env.get_state().screen_buffer
                next_state = stack_frame_fn(frames_queue, next_frame)
                state = next_state

        score = env.get_total_reward()
        print("Score: ", score)
    env.close()

