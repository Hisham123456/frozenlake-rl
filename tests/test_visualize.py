import json

import gymnasium as gym
import numpy as np
import pytest

from src.agent import QLearningAgent
from src.train import train_agent
from src.visualize import build_history, replay_policy, save_history


def small_history():
    return build_history(n_episodes=30, checkpoints=[10], eval_episodes=5,
                         n_replays=3, eval_seed=20000)


def test_history_is_reproducible_and_contains_checkpoints():
    first = small_history()
    assert first == small_history()
    assert [cp["episode"] for cp in first["checkpoints"]] == [0, 10, 30]
    for cp in first["checkpoints"]:
        assert np.asarray(cp["q_table"]).shape == (16, 4)
        assert len(cp["evaluation"]["trials"]) == 3
        assert len(cp["exploration_trials"]) == 3
        assert cp["evaluation"]["n_episodes"] == 5


def test_checkpoints_do_not_change_training_results():
    history = small_history()
    env = gym.make("FrozenLake-v1", is_slippery=True)
    agent = QLearningAgent(16, 4, seed=42)
    try:
        train_agent(env, agent, 30, 0.1, 0.99, 0.3, seed=42)
    finally:
        env.close()
    np.testing.assert_array_equal(history["checkpoints"][-1]["q_table"], agent.q_table)


def test_replays_are_real_transitions_and_do_not_mutate_table():
    table = np.zeros((16, 4))
    before = table.copy()
    replay = replay_policy(table, 10, 42, 20000, keep=10)
    env = gym.make("FrozenLake-v1", is_slippery=True)
    try:
        for trial in replay["trials"]:
            state, _ = env.reset(seed=trial["seed"])
            assert state == trial["initial_state"]
            for transition in trial["transitions"]:
                assert transition["from"] == state
                state, reward, terminated, truncated, _ = env.step(transition["action"])
                assert transition["to"] == state
                assert transition["reward"] == reward
            assert terminated or truncated
            assert trial["outcome"] == ("success" if trial["reward"] else (
                "hole" if terminated else "timeout"))
    finally:
        env.close()
    np.testing.assert_array_equal(table, before)


def test_export_is_standalone_and_checkpoints_reload(tmp_path):
    history = small_history()
    page = save_history(history, tmp_path)
    html = page.read_text(encoding="utf-8")
    assert "__LEARNING_DATA__" not in html
    assert '<html lang="fr">' in html
    assert "fetch(" not in html
    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8")) == history
    for cp in history["checkpoints"]:
        loaded = np.load(tmp_path / f"q_table_{cp['episode']:06d}.npy")
        np.testing.assert_array_equal(loaded, cp["q_table"])


@pytest.mark.parametrize("kwargs", [
    {"eval_seed": 50}, {"eval_episodes": 0}, {"n_replays": 201},
    {"checkpoints": [-1]}, {"checkpoints": [10001]}, {"epsilon": 1.1},
])
def test_visualization_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        build_history(**kwargs)
