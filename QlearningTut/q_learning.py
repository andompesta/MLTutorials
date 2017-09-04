import numpy as np


class QLearning():
    def __init__(self, actions, learning_rate=0.1, epsilon=0.9, gamma=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.

    def choose_action(self, state):
        rand = np.random.random_sample()
        if rand < self.epsilon:
            # execute greedy action

        else:
            # execute random action

