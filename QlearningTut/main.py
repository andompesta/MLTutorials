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

from QlearningTut.maze_env import Maze
# from QlearningTut.q_learning import QLearning, TITLE
from QlearningTut.sarsa import Sarsa, TITLE
import pickle
from os.path import join as join_path

MAZE_H = 4
MAZE_W = 8

def fix_state(state):
    state = [int((state[1]-5)/40), int((state[0]-5)/40)]
    return state


def print_q_table(table):
    for y in range(MAZE_H):
        for x in range(MAZE_W):
            print("{}-{}\t{}".format(x, y, " ".join(map(str, table[y, x, :]))) )


def dump(RL_algo, file_name, path="."):
    with open(join_path(path, '{}.pkl'.format(file_name)), 'wb') as output:
        pickle.dump(RL_algo, output, pickle.HIGHEST_PROTOCOL)


def update():
    episode = 0
    while True:
        # initial observation
        state = fix_state(env.reset())
        action = RL.choose_action(state)
        while True:
            # fresh env
            env.render()

            state_, reward, done = env.step(action)
            state_ = fix_state(state_)
            action_ = RL.choose_action(state_)

            if TITLE == "Q-learning":
                RL.learning(state, action, reward, state_, done)
            else:
                RL.learning(state, action, reward, state_, action_, done)

            state = state_
            action = action_


            # break while loop when end of this episode
            if done:
                episode += 1
                print("\n\n---------{}\n\n".format(episode))
                print_q_table(RL.q_table)
                print("reward: {}".format(reward))
                break
        dump(RL, TITLE)

    env.destroy()

if __name__ == "__main__":
    env = Maze(TITLE, height=MAZE_H, width=MAZE_W)

    # RL = QLearning(actions=list(range(env.n_actions)), env_size=(MAZE_H, MAZE_W))
    RL = Sarsa(actions=list(range(env.n_actions)), env_size=(MAZE_H, MAZE_W))
    env.after(100, update)
    env.mainloop()