"""Evaluate random and saved greedy policies on slippery 4x4 FrozenLake."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.agent import RandomAgent


class GreedyAgent:
    """Frozen policy: choose the first maximum on ties; never learn."""

    def __init__(self, q_table):
        table = np.asarray(q_table)
        if table.shape != (16, 4):
            raise ValueError("Q-table must have shape (16, 4)")
        if not np.issubdtype(table.dtype, np.number) or np.iscomplexobj(table):
            raise ValueError("Q-table must contain real numbers")
        if not np.isfinite(table).all():
            raise ValueError("Q-table must contain only finite values")
        self.q_table = table.copy()
        self.q_table.flags.writeable = False

    def select_action(self, state):
        return int(np.argmax(self.q_table[state]))


def load_q_table(path):
    try:
        table = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"Cannot load Q-table '{path}': {error}") from error
    if not isinstance(table, np.ndarray):
        table.close()
        raise ValueError("Q-table must be a .npy array")
    return GreedyAgent(table)


def run_episode(env, agent, seed=None):
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        # Preserve the original RandomAgent API.
        action = agent.select_action() if isinstance(agent, RandomAgent) else agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    return total_reward, steps


def evaluate_agent(env, agent, n_episodes=100, seed=42):
    if n_episodes <= 0:
        raise ValueError("n_episodes must be greater than zero")
    successes = 0
    total_reward = 0.0
    total_steps = 0
    for episode in range(n_episodes):
        reward, steps = run_episode(env, agent, seed=seed + episode)
        successes += int(reward > 0)
        total_reward += reward
        total_steps += steps
    return {
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "average_reward": total_reward / n_episodes,
        "average_steps": total_steps / n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["random", "q-learning", "compare"], default="random")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--q-table", type=Path, default=Path("results/q_learning/q_table.npy"))
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    if args.agent_seed < 0 or args.env_seed < 0:
        parser.error("seeds must be non-negative")
    agents = {}
    if args.agent in ("random", "compare"):
        agents["random"] = RandomAgent(n_actions=4, seed=args.agent_seed)
    if args.agent in ("q-learning", "compare"):
        try:
            agents["q-learning"] = load_q_table(args.q_table)
        except ValueError as error:
            parser.error(str(error))
    report = {
        "environment": "FrozenLake-v1",
        "map_name": "4x4",
        "is_slippery": True,
        "max_episode_steps": 100,
        "n_episodes": args.episodes,
        "env_seed_start": args.env_seed,
        "env_seed_end": args.env_seed + args.episodes - 1,
        "random_agent_seed": args.agent_seed if "random" in agents else None,
        "greedy_policy": "epsilon=0; first maximum on ties; no learning",
        "gymnasium_version": gym.__version__,
        "numpy_version": np.__version__,
        "results": {},
    }
    if "q-learning" in agents:
        report["q_table_path"] = args.q_table.as_posix()
        report["q_table_sha256"] = hashlib.sha256(args.q_table.read_bytes()).hexdigest()
        parameters_path = args.q_table.with_name("parameters.json")
        if parameters_path.exists():
            try:
                report["training_parameters"] = json.loads(parameters_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                parser.error(f"Cannot read training parameters: {error}")
    for name, agent in agents.items():
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, max_episode_steps=100)
        try:
            results = evaluate_agent(env, agent, args.episodes, args.env_seed)
        finally:
            env.close()
        report["results"][name] = results
        print(f"Agent: {name}")
        print(f"Episodes: {args.episodes}")
        print(f"Environment seeds: {report['env_seed_start']} to {report['env_seed_end']}")
        if name == "random":
            print(f"Agent seed: {args.agent_seed}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Average reward: {results['average_reward']:.4f}")
        print(f"Average episode length: {results['average_steps']:.2f}")
    if args.output:
        protected = {args.q_table.resolve(), args.q_table.with_name("parameters.json").resolve()}
        if args.output.resolve() in protected:
            parser.error("Report destination must differ from training artifacts")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
