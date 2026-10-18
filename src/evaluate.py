import argparse

import gymnasium as gym

from src.agent import RandomAgent

def run_episode(env, agent, seed=None):
    state, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.select_action()

        state, reward, terminated, truncated, info = env.step(action)

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
        reward, steps = run_episode(
            env,
            agent,
            seed=seed + episode,
        )

        if reward > 0:
            successes += 1

        total_reward += reward
        total_steps += steps

    return {
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "average_reward": total_reward / n_episodes,
        "average_steps": total_steps / n_episodes,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a random agent on FrozenLake."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--env-seed", type=int, default=42)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,
    )

    try:
        agent = RandomAgent(
            n_actions=env.action_space.n,
            seed=args.agent_seed,
        )
        results = evaluate_agent(
            env,
            agent,
            n_episodes=args.episodes,
            seed=args.env_seed,
        )

        print(f"Episodes: {results['n_episodes']}")
        print(f"Agent seed: {args.agent_seed}")
        print(f"Environment seed: {args.env_seed}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Average reward: {results['average_reward']:.4f}")
        print(f"Average episode length: {results['average_steps']:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()