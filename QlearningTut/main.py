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
from QlearningTut.q_learning import QLearning
import numpy as np

def fix_state(state):
    state = [int((state[0]-5)/40), int((state[1]-5)/40)]
    return state


def update():
    for episode in range(100):
        # initial observation
        state = fix_state(env.reset())

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(state)

            # RL take action and get next observation and reward
            state_, reward, done = env.step(action)
            state_ = fix_state(state_)
            # RL learn from this transition
            RL.learning(state, action, reward, state_, done)

            # swap observation
            state = state_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearning(actions=list(range(env.n_actions)), env_size=(4,4))
    env.after(100, update)
    env.mainloop()