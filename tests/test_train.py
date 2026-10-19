import gymnasium as gym
import numpy as np

from src.agent import QLearningAgent
from src.train import train_agent


def test_training_is_reproducible():
    def run_training():
        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=True,
        )

        try:
            agent = QLearningAgent(
                n_states=env.observation_space.n,
                n_actions=env.action_space.n,
                seed=42,
            )

            train_agent(
                env,
                agent,
                n_episodes=1000,
                alpha=0.1,
                gamma=0.99,
                epsilon=0.3,
                seed=42,
            )

            return agent.q_table.copy()
        finally:
            env.close()


    first_table = run_training()
    second_table = run_training()

    np.testing.assert_array_equal(first_table, second_table)
    assert np.any(first_table > 0)