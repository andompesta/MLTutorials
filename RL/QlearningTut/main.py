"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import pickle
from os.path import join as join_path

import imageio
import numpy as np
from RL.envs.maze_env_gif import Maze

# from QlearningTut.q_learning import QLearning, TITLE
from RL.QlearningTut.sarsa import Sarsa, TITLE

MAZE_H = 4
MAZE_W = 8

def fix_state(state):
    state = [int((state[1]-5)/40), int((state[0]-5)/40)]
    return state


def print_q_table(table):
    for x in range(MAZE_W):
        for y in range(MAZE_H):
            print("{}-{}\t{}".format(x, y, " ".join(map(str, table[x, y, :]))) )


def dump(RL_algo, file_name, path="."):
    with open(join_path(path, '{}.pkl'.format(file_name)), 'wb') as output:
        pickle.dump(RL_algo, output, pickle.HIGHEST_PROTOCOL)


def update():
    episode = 0
    while True:
        # initial observation
        state = env.reset()
        action = RL.choose_action(state)
        frames = [np.array(env.render(0))]

        for step in range(100):

            state_, reward, done = env.step(action)
            action_ = RL.choose_action(state_)

            if action_ >= 4:
                print_q_table(RL.q_table)
                RL.choose_action(state_)
                print(state_)
                print(action_)

            if TITLE == "Q-learning":
                RL.learning(state, action, reward, state_, done)
            else:
                RL.learning(state, action, reward, state_, action_, done)

            state = state_
            action = action_
            frames.append(np.array(env.render(step+1)))


            # break while loop when end of this episode
            if done:
                print("\n\n---------{}\n\n".format(episode))
                print_q_table(RL.q_table)
                print("reward: {}".format(reward))
                break

        if episode % 10 == 0:
            time_per_step = 0.25
            images = np.array(frames)
            image_file = join_path(".", "image", '{}-{}.gif'.format(TITLE, episode))
            imageio.mimsave(image_file, images, duration=time_per_step)
            dump(RL, '{}'.format(TITLE))
        episode += 1
if __name__ == "__main__":
    env = Maze(height=MAZE_H, width=MAZE_W)
    # RL = QLearning(actions=list(range(env.n_actions)), env_size=(MAZE_W, MAZE_H))
    RL = Sarsa(actions=list(range(env.n_actions)), env_size=(MAZE_W, MAZE_H))
    update()