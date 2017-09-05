import numpy as np


class QLearning:
    def __init__(self, actions, env_size, learning_rate=0.1, epsilon=0.9, gamma=0.9):
        self.actions = actions
        self.env_size = env_size
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        q_table_size = env_size + (len(self.actions),)
        self.q_table = np.zeros(q_table_size)


    def choose_action(self, state):
        rand = np.random.uniform(0, 1)
        if rand < self.epsilon:
            # execute greedy action
            state_values = self.q_table[state[0], state[1], :]
            return np.random.choice(np.argwhere(state_values == np.amax(state_values)).flatten())
        else:
            # execute random action
            return np.random.choice(self.q_table[state[0], state[1], :]).astype(np.int64)


    def learning(self, state, action, reward, state_prime, done):
        if done:
            self.q_table[state[0], state[1], action] += self.lr * (reward - self.q_table[state[0], state[1], action])
        else:
            self.q_table[state[0], state[1], action] += self.lr * (reward + (self.gamma*np.max(self.q_table[state_prime[0], state_prime[1], :])) - self.q_table[state[0], state[1], action])
