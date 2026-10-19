from src.agent import RandomAgent


def test_random_agent_returns_valid_actions():
    agent = RandomAgent(n_actions=4, seed=42)

    actions = [agent.select_action() for _ in range(100)]

    assert all(action in range(4) for action in actions)


def test_random_agent_is_reproducible():
    first_agent = RandomAgent(n_actions=4, seed=42)
    second_agent = RandomAgent(n_actions=4, seed=42)

    first_actions = [first_agent.select_action() for _ in range(20)]
    second_actions = [second_agent.select_action() for _ in range(20)]

    assert first_actions == second_actions