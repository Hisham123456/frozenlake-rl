"""Train checkpoints and generate an offline FrozenLake learning viewer."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.agent import QLearningAgent
from src.train import train_agent


def replay_policy(q_table, n_episodes, agent_seed, env_seed, epsilon=0, keep=12):
    """Evaluate a copied table using independent environments and RNGs."""
    policy = QLearningAgent(16, 4, seed=agent_seed)
    policy.q_table = q_table.copy()
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True,
                   max_episode_steps=100)
    trials = []
    successes = total_steps = 0
    total_reward = 0.0
    try:
        for episode in range(n_episodes):
            state, _ = env.reset(seed=env_seed + episode)
            initial_state = int(state)
            transitions = []
            reward_sum = 0.0
            while True:
                action = policy.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                if episode < keep:
                    transitions.append({
                        "from": int(state), "to": int(next_state),
                        "action": action, "reward": float(reward),
                    })
                reward_sum += reward
                total_steps += 1
                state = next_state
                if terminated or truncated:
                    break
            successes += int(reward_sum > 0)
            total_reward += reward_sum
            if episode < keep:
                trials.append({
                    "seed": env_seed + episode, "initial_state": initial_state,
                    "transitions": transitions, "reward": float(reward_sum),
                    "outcome": "success" if reward_sum > 0 else (
                        "hole" if terminated else "timeout"),
                })
    finally:
        env.close()
    return {
        "success_rate": successes / n_episodes,
        "average_reward": total_reward / n_episodes,
        "average_steps": total_steps / n_episodes,
        "successes": successes, "n_episodes": n_episodes,
        "trials": trials,
    }


def build_history(n_episodes=10000, checkpoints=None, eval_episodes=200,
                  n_replays=12, alpha=0.1, gamma=0.99, epsilon=0.3,
                  agent_seed=42, env_seed=42, eval_seed=20000):
    if n_episodes <= 0 or eval_episodes <= 0 or not 1 <= n_replays <= eval_episodes:
        raise ValueError("Episode counts must be positive; replays must not exceed evaluation episodes")
    if min(agent_seed, env_seed, eval_seed) < 0:
        raise ValueError("Seeds must be non-negative")
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= epsilon <= 1:
        raise ValueError("Invalid alpha, gamma, or epsilon")
    if max(env_seed, eval_seed) <= min(env_seed + n_episodes - 1, eval_seed + eval_episodes - 1):
        raise ValueError("Training and evaluation seed ranges must not overlap")
    if checkpoints is None:
        checkpoints = [100, 500, 1000, 2500, 5000]
        checkpoints = [point for point in checkpoints if point < n_episodes]
    if any(point < 0 or point > n_episodes for point in checkpoints):
        raise ValueError("Checkpoints must lie between zero and the training episode count")
    checkpoints = sorted(set([0, n_episodes, *checkpoints]))
    history = {
        "schema_version": 1,
        "map": ["SFFF", "FHFH", "FFFH", "HFFG"],
        "parameters": {
            "n_episodes": n_episodes, "alpha": alpha, "gamma": gamma,
            "epsilon": epsilon, "agent_seed": agent_seed, "env_seed": env_seed,
            "eval_seed": eval_seed, "eval_episodes": eval_episodes,
            "n_replays": n_replays, "max_episode_steps": 100,
            "gymnasium_version": gym.__version__, "numpy_version": np.__version__,
            "greedy_ties": "seeded random maximum; epsilon=0",
        },
        "checkpoints": [],
    }
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True,
                   max_episode_steps=100)
    agent = QLearningAgent(16, 4, seed=agent_seed)

    def capture(episode, current_agent):
        if episode not in checkpoints:
            return
        table = current_agent.q_table.copy()
        evaluation = replay_policy(table, eval_episodes, agent_seed, eval_seed,
                                   keep=n_replays)
        exploration = replay_policy(table, n_replays, agent_seed, eval_seed,
                                    epsilon=epsilon, keep=n_replays)
        history["checkpoints"].append({
            "episode": episode, "q_table": table.tolist(),
            "evaluation": evaluation, "exploration_trials": exploration["trials"],
        })
        print(f"Checkpoint {episode}: success {evaluation['success_rate']:.1%}")

    try:
        capture(0, agent)
        train_agent(env, agent, n_episodes, alpha, gamma, epsilon,
                    seed=env_seed, on_episode_end=capture)
    finally:
        env.close()
    return history


def save_history(history, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for checkpoint in history["checkpoints"]:
        np.save(output / f"q_table_{checkpoint['episode']:06d}.npy",
                np.asarray(checkpoint["q_table"]))
    data = json.dumps(history, ensure_ascii=False, allow_nan=False)
    (output / "history.json").write_text(data + "\n", encoding="utf-8")
    template = Path(__file__).with_name("learning_viewer.html").read_text(encoding="utf-8")
    # Prevent user-supplied strings in exported data from closing the script tag.
    safe_data = data.replace("<", "\\u003c").replace("&", "\\u0026")
    page = template.replace("__LEARNING_DATA__", safe_data)
    destination = output / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/learning"))
    args = parser.parse_args()
    try:
        history = build_history(
            args.episodes, args.checkpoints, args.eval_episodes, args.replays,
            args.alpha, args.gamma, args.epsilon, args.agent_seed, args.env_seed,
            args.eval_seed,
        )
    except ValueError as error:
        parser.error(str(error))
    destination = save_history(history, args.output_dir)
    print(f"Open in your browser: {destination.resolve()}")


if __name__ == "__main__":
    main()
