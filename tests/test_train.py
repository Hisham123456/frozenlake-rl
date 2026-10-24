import gymnasium as gym
import numpy as np
import json
import pytest

from src.agent import QLearningAgent
from src.train import save_training, train_agent




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

def test_save_training_preserves_table_and_parameters(tmp_path):
    agent = QLearningAgent(n_states=16, n_actions=4)
    agent.q_table[14, 2] = 0.5

    parameters = {
        "n_episodes": 1000,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.3,
        "seed": 42,
    }

    output_dir = tmp_path / "training"
    save_training(agent, output_dir, parameters)

    loaded_table = np.load(output_dir / "q_table.npy")

    with (output_dir / "parameters.json").open(
        encoding="utf-8"
    ) as file:
        loaded_parameters = json.load(file)

    np.testing.assert_array_equal(loaded_table, agent.q_table)
    assert loaded_parameters == parameters

@pytest.mark.parametrize(
    "parameter, value",
    [
        ("n_episodes", 0),
        ("n_episodes", -1),
        ("alpha", 0),
        ("alpha", -0.1),
        ("alpha", 1.1),
        ("gamma", -0.1),
        ("gamma", 1.1),
        ("epsilon", -0.1),
        ("epsilon", 1.1),
    ],
)
def test_training_rejects_invalid_parameters(parameter, value):
    parameters = {
        "n_episodes": 100,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.3,
    }
    parameters[parameter] = value

    with pytest.raises(ValueError):
        train_agent(env=None, agent=None, **parameters)