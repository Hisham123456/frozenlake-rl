import json
from pathlib import Path

import numpy as np
import argparse

import gymnasium as gym

from src.agent import QLearningAgent


def train_agent(
    env, agent, n_episodes, alpha, gamma, epsilon, seed=42,
    on_episode_end=None,
):
    if n_episodes <= 0:
        raise ValueError("n_episodes must be greater than zero")
    
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    
    for episode in range(n_episodes):
        state, info = env.reset(seed=seed + episode)

        while True:
            action = agent.select_action(state, epsilon)

            next_state, reward, terminated, truncated, info = env.step(action)

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                alpha=alpha,
                gamma=gamma,
            )

            state = next_state

            if terminated or truncated:
                break

        if on_episode_end is not None:
            on_episode_end(episode + 1, agent)

def save_training(agent, output_dir, parameters):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "q_table.npy", agent.q_table)

    with (output_path / "parameters.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(parameters, file, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Train a Q-Learning agent on FrozenLake."
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/q_learning")
    args = parser.parse_args()

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,
    )

    try:
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            seed=args.agent_seed,
        )

        train_agent(
            env,
            agent,
            n_episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=args.env_seed,
        )

        parameters = {
            "environment": "FrozenLake-v1",
            "map_name": "4x4",
            "is_slippery": True,
            "n_episodes": args.episodes,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "agent_seed": args.agent_seed,
            "env_seed": args.env_seed,
            "gymnasium_version": gym.__version__,
            "numpy_version": np.__version__,
        }
        save_training(agent, args.output_dir, parameters)

        print(f"Training completed: {args.episodes} episodes")
        print(f"Results saved to: {Path(args.output_dir).resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
