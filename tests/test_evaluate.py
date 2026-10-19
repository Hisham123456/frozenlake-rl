import gymnasium as gym

from src.agent import RandomAgent
from src.evaluate import evaluate_agent


def test_evaluation_is_reproducible():
    def run_evaluation():
        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=True,
        )

        try:
            agent = RandomAgent(n_actions=env.action_space.n, seed=42)
            return evaluate_agent(env, agent, n_episodes=100, seed=42)
        finally:
            env.close()

    assert run_evaluation() == run_evaluation()