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