import numpy as np


class RandomAgent:
    def __init__(self, n_actions, seed=None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
    
    def select_action(self):
        return int(self.rng.integers(self.n_actions))