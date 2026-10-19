def train_agent(env, agent, n_episodes, alpha, gamma, epsilon, seed=42):
    if n_episodes <= 0:
        raise ValueError("n_episodes must be greater than zero")

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