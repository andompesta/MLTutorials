import argparse
import random                # Handling random number generation
import time
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
from RL.DQN.model_dqn import DQN, epsilon_greedy_policy

from visdom import Visdom
from RL.helper import frame_processor
import RL.helper as helper
import itertools
import matplotlib.pyplot as plt
import gym
from RL.envs.maze_env_gif import Maze
from collections import deque

EXP_NAME = "exp-dqn"
GLOBAL_STEP = 0
P_LOSSES = []
RESET_FRAME = torch.zeros([1, 4, 34, 136])
HISTOGRAM = {0:0, 1:0, 2:0, 3:0}


def __pars_args__():
    parser = argparse.ArgumentParser(description='DQN')

    parser.add_argument('--max_grad', type=float, default=1, help='value loss coefficient (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size used during learning')

    parser.add_argument('-m_path', '--model_path', default='./model', help='Path to save the model')
    parser.add_argument('-v_path', '--monitor_path', default='./video', help='Path to save videos of agent')

    parser.add_argument("-u_target", "--update_target_estimator_every", default=8,
                        help="how ofter update the parameters of the target network")
    parser.add_argument("-ne", "--num_episodes", type=int, default=5000, help="Number of episodes to run for")
    # parser.add_argument('-a', '--actions', type=list, default=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], help='possible actions')
    parser.add_argument('-a', '--actions', type=list, default=[0, 1, 2, 3],
                        help='possible actions')
    parser.add_argument('-nf', '--number_frames', type=int, default=4, help='number of frame for each state')
    parser.add_argument('-ds', '--discount_factor', type=float, default=0.99, help='Reward discount factor')
    parser.add_argument("-es", "--epsilon_start", type=float, default=0.9, help="starting epsilon")
    parser.add_argument("-ee", "--epsilon_end", type=float, default=0.05, help="ending epsilon")
    parser.add_argument("-ed", "--epsilon_decay_rate", type=float, default=0.005,
                        help="Number of steps to decay epsilon over")
    parser.add_argument("-rv", "--record_video_every", type=int, default=50, help="Record a video every N episodes")
    parser.add_argument("-rm", "--replay_memory_size", type=int, default=10000, help="Size of the replay memory")
    parser.add_argument("-rm_init", "--replay_memory_init_size", type=int, default=500,
                        help="Number of random experiences to sample when initializing the reply memory")
    parser.add_argument("--max_steps", type=int, default=100, help="Max step for an episode")
    parser.add_argument("--state_size", type=list, default=[34, 136], help="Frame size")
    parser.add_argument("-uc", "--use_cuda", type=bool, default=False, help="Use cuda")
    parser.add_argument("-v", "--version", type=str, default="v-0", help="Use cuda")

    return parser.parse_args()




def create_enviroment():
    # env = vz.DoomGame()
    # env.load_config(path.join("../", "doom_setup", "doom_basic.cfg"))
    # env.set_doom_scenario_path(path.join("../", "doom_setup", "basic.wad"))
    # env.set_window_visible(False)
    # env.init()

    env = Maze(height=4, width=8)
    env.init(True)

    # env = gym.make('CartPole-v0')
    # env = env.unwrapped
    return env

def test_environment(args):
    game = create_enviroment()

    crop = (50, 10, 30, 30)
    img_trans = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(args.state_size),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            frame = np.expand_dims(state.screen_buffer, axis=-1)
            misc = state.game_variables
            frame = frame_processor(frame, crop, img_trans)

            plt.figure()
            plt.imshow(frame[:,:, 0].numpy())
            plt.show()

            action = random.choice(args.actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()





if __name__ == '__main__':
    args = __pars_args__()

    IMG_TRANS = T.Compose([T.Resize(args.state_size),
                           T.Grayscale(),
                           T.ToTensor()])

    p_rewards = []



    # test_environment(args)
    env = create_enviroment()
    device = torch.device("cuda:0" if args.use_cuda else "cpu")


    vis = Visdom()
    q_network = DQN()
    t_network = DQN()

    q_network.to(device)
    t_network.to(device)

    optimizer = torch.optim.RMSprop(q_network.parameters())
    t_network.eval()

    replay_memory = helper.ExperienceBuffer(args.replay_memory_size)
    stack_frame_fn = helper.stack_frame_setup(IMG_TRANS)
    frames_queue = deque(maxlen=args.number_frames)


    def optimize_model():
        if len(replay_memory.buffer) < args.batch_size:
            return

        global GLOBAL_STEP

        batch = replay_memory.sample(args.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)


        state_batch = torch.cat(state_batch).to(device)
        action_batch = torch.tensor(action_batch).long().to(device)
        reward_batch = torch.tensor(reward_batch).to(device)
        done_batch = torch.tensor(done_batch).to(device)
        next_state_batch = torch.cat(next_state_batch).to(device)


        with torch.no_grad():
            non_final_mask = (1 - done_batch).byte().to(device)
            non_final_next_states = next_state_batch[non_final_mask]

            next_state_values = torch.zeros(args.batch_size, device=device)
            next_state_values[non_final_mask] = t_network(non_final_next_states).max(1)[0]
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * args.discount_factor) + reward_batch

        state_action_values = q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        P_LOSSES.append([loss.item(), state_action_values.max().item()])
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(q_network.parameters(), args.max_grad)
        optimizer.step()
        assert t_network.head.bias.grad == None
        assert expected_state_action_values.grad_fn == None
        GLOBAL_STEP += 1


    print("record_video_every:{}\treplay_memory_size:{}\treplay_memory_init_size:{}".format(args.record_video_every,
                                                                                            args.replay_memory_size,
                                                                                            args.replay_memory_init_size))



    policy = epsilon_greedy_policy(q_network,
                                   args.epsilon_end,
                                   args.epsilon_start,
                                   args.epsilon_decay_rate,
                                   args.actions, device)

    for i_episode in range(args.num_episodes):
        # env.reset()
        # last_screen = helper.get_screen(env, device)
        # current_screen = helper.get_screen(env, device)
        # state = current_screen - last_screen
        # assert state.sum() == 0

        env.new_episode()
        frame = env.render()
        total_reward = 0
        state = stack_frame_fn(frames_queue, frame, True)

        # One step in the environment
        for t in itertools.count():
            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}".format(t, GLOBAL_STEP, i_episode + 1), end="")


            # Observe new state
            action, epsilon = policy(state, GLOBAL_STEP)
            HISTOGRAM[action] += 1
            # _, reward, done, _ = env.step(action.item())
            # total_reward += reward
            reward = env.make_action(action)
            total_reward += reward
            done = env.is_episode_finished()

            if done:
                transition = helper.Transition(state, action, RESET_FRAME, reward, int(done))
                replay_memory.store(transition)

                # start new episode
                env.new_episode()
                # frame = env.get_state().screen_buffer
                frame = env.render()
                state = stack_frame_fn(frames_queue, frame, True)
            else:
                # frame = env.get_state().screen_buffer
                frame = env.render()

                next_state = stack_frame_fn(frames_queue, frame)
                transition = helper.Transition(state, action, next_state, reward, int(done))
                replay_memory.store(transition)
                state = next_state


            optimize_model()

            if (t == args.max_steps) or done:
                stat = helper.EpisodeStat(t + 1, total_reward)
                print('Episode: {}'.format(i_episode),
                      'Episode length: {}'.format(stat.episode_length),
                      'Episode reward: {}'.format(stat.episode_reward),
                      'Avg reward: {}'.format(stat.avg_reward))
                break

        if len(P_LOSSES) > 0:
            vis.line(Y=np.array(P_LOSSES),
                     X=np.repeat(np.expand_dims(np.arange(len(P_LOSSES)), 1), 2, axis=1),
                     opts=dict(legend=["loss", "max_q_value"],
                               title="q_network",
                               showlegend=True),
                     win="plane_q_network_{}".format(EXP_NAME))
            print(HISTOGRAM)

        # Update params
        if i_episode % args.update_target_estimator_every == 0:
            t_network.load_state_dict(q_network.state_dict())
            t_network.eval()
            print("\nCopied model parameters to target network.")
            for t_param, q_param in zip(t_network.state_dict().items(), q_network.state_dict().items()):
                if not torch.equal(t_param[1], q_param[1]):
                    print("Error : {}\t{}".format(t_param[0], q_param[0]))
                    break


        p_rewards.append([stat.episode_reward, stat.episode_length, stat.avg_reward])
        vis.line(Y=np.array(p_rewards),
                 X=np.repeat(np.expand_dims(np.arange(i_episode + 1), 1), 3, axis=1),
                 opts=dict(legend=["episode_reward", "episode_length", "average_reward"],
                           title="rewards",
                           showlegend=True),
                 win="plane_reward_{}".format(EXP_NAME))
        
    env.close()
