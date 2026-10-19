import numpy as np


class RandomAgent:
    def __init__(self, n_actions, seed=None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
    
    def select_action(self):
        return int(self.rng.integers(self.n_actions))
    

class QLearningAgent:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))

        values = self.q_table[state]
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))
    
    def update(self, state, action, reward, next_state, terminated, alpha, gamma):
        target = reward

        if not terminated:
            target += gamma * self.q_table[next_state].max()

        current_value = self.q_table[state, action]

        self.q_table[state, action] += alpha * (target - current_value)