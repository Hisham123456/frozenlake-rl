from src.agent import RandomAgent
from src.agent import QLearningAgent


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

def test_q_learning_table_starts_with_zeros():
    agent = QLearningAgent(n_states=16, n_actions=4)

    assert agent.q_table.shape == (16, 4)
    assert (agent.q_table == 0).all()

def test_q_learning_greedy_action():
    agent = QLearningAgent(n_states=16, n_actions=4, seed=42)
    agent.q_table[0] = [0.1, 0.2, 0.9, 0.3]

    action = agent.select_action(state=0, epsilon=0)

    assert action == 2

def test_q_learning_explores_with_epsilon_one():
    agent = QLearningAgent(n_states=16, n_actions=4, seed=42)
    agent.q_table[0] = [0.1, 0.2, 0.9, 0.3]

    actions = [
        agent.select_action(state=0, epsilon=1)
        for _ in range(100)
    ]

    assert all(action in range(4) for action in actions)
    assert any(action != 2 for action in actions)

def test_q_learning_update_non_terminal_transition():
    agent = QLearningAgent(n_states=16, n_actions=4)
    agent.q_table[0, 1] = 0.1
    agent.q_table[4] = [0.2, 0.5, 0.3, 0.1]

    agent.update(
        state=0,
        action=1,
        reward=0,
        next_state=4,
        terminated=False,
        alpha=0.5,
        gamma=0.9,
    )

    assert abs(agent.q_table[0, 1] - 0.275) < 1e-10

def test_q_learning_update_terminal_transition():
    agent = QLearningAgent(n_states=16, n_actions=4)
    agent.q_table[15] = [10, 10, 10, 10]

    agent.update(
        state=14,
        action=2,
        reward=1,
        next_state=15,
        terminated=True,
        alpha=0.5,
        gamma=0.9,
    )

    assert agent.q_table[14, 2] == 0.5