import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

import src.evaluate as evaluation
from src.agent import RandomAgent
from src.evaluate import GreedyAgent, evaluate_agent, load_q_table, run_episode


@pytest.mark.parametrize("table", [np.zeros((4, 4)), np.full((16, 4), np.nan),
                                  np.full((16, 4), np.inf), np.full((16, 4), "x"),
                                  np.ones((16, 4), dtype=complex)])
def test_rejects_invalid_q_tables(table):
    with pytest.raises(ValueError):
        GreedyAgent(table)


def test_load_q_table(tmp_path):
    table = np.zeros((16, 4))
    table[0, 2] = 1
    path = tmp_path / "table.npy"
    np.save(path, table)
    agent = load_q_table(path)
    assert agent.select_action(0) == 2
    np.testing.assert_array_equal(agent.q_table, table)


def test_missing_and_corrupt_tables(tmp_path):
    with pytest.raises(ValueError, match="Cannot load"):
        load_q_table(tmp_path / "missing.npy")
    path = tmp_path / "corrupt.npy"
    path.write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load"):
        load_q_table(path)


def test_evaluation_does_not_modify_q_table():
    table = np.arange(64, dtype=float).reshape(16, 4)
    before = table.copy()
    agent = GreedyAgent(table)
    env = gym.make("FrozenLake-v1", is_slippery=True)
    try:
        first = evaluate_agent(env, agent, 20, seed=20000)
        second = evaluate_agent(env, agent, 20, seed=20000)
    finally:
        env.close()
    assert first == second
    np.testing.assert_array_equal(agent.q_table, before)
    np.testing.assert_array_equal(table, before)


def test_greedy_ties_choose_first_maximum():
    table = np.zeros((16, 4))
    table[0] = [0, 1, 1, 0]
    assert GreedyAgent(table).select_action(0) == 1


def test_metrics_and_episode_seeds(monkeypatch):
    outcomes = iter([(1.0, 6), (0.0, 2)])
    seeds = []

    def episode(env, agent, seed):
        seeds.append(seed)
        return next(outcomes)

    monkeypatch.setattr(evaluation, "run_episode", episode)
    result = evaluate_agent(None, None, n_episodes=2, seed=20000)
    assert seeds == [20000, 20001]
    assert result == {"n_episodes": 2, "success_rate": 0.5,
                      "average_reward": 0.5, "average_steps": 4.0}


@pytest.mark.parametrize("episodes", [0, -1])
def test_rejects_invalid_episode_count(episodes):
    with pytest.raises(ValueError):
        evaluate_agent(None, None, episodes)


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True)])
def test_episode_stops_for_both_endings(terminated, truncated):
    class SingleStepEnv:
        def reset(self, seed=None):
            self.calls = 0
            return 0, {}

        def step(self, action):
            self.calls += 1
            assert self.calls == 1
            return 4, 0, terminated, truncated, {}

    assert run_episode(SingleStepEnv(), RandomAgent(4, seed=42)) == (0.0, 1)


def test_comparison_cli_saves_reproducible_report(tmp_path):
    table_path = tmp_path / "table.npy"
    np.save(table_path, np.zeros((16, 4)))
    output = tmp_path / "comparison.json"
    command = [sys.executable, "-m", "src.evaluate", "--agent", "compare",
               "--episodes", "10", "--env-seed", "20000", "--q-table", str(table_path),
               "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first = json.loads(output.read_text(encoding="utf-8"))
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert set(first["results"]) == {"random", "q-learning"}
    assert first["env_seed_start"] == 20000
    assert first["env_seed_end"] == 20009


def test_cli_missing_table_has_clear_error(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--agent", "q-learning",
         "--q-table", str(tmp_path / "missing.npy")],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Cannot load Q-table" in result.stderr
