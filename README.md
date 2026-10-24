# FrozenLake Reinforcement Learning

A Python project for learning and experimenting with reinforcement learning using the FrozenLake environment from Gymnasium.

## Project status

The project is currently in its initial setup phase. The reinforcement learning agents will be implemented progressively through dedicated GitHub Issues.

## Requirements

- Python 3.14
- Git

## Installation

1. Clone the repository:

```powershell
git clone https://github.com/Hisham123456/frozenlake-rl.git
cd frozenlake-rl
```

2. Create a virtual environment:

```powershell
python -m venv .venv
```

3. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow locally created scripts for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

4. Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Current structure

```text
frozenlake-rl/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── evaluate.py
│   └── train.py
├── tests/
│   ├── test_agent.py
│   ├── test_evaluate.py
│   └── test_train.py
└── notebooks/
    └── exploration.ipynb  
```
## Random agent baseline

The random agent selects each of the four actions with equal probability.
It does not learn from observations or rewards.

Run the evaluation from the project root, with the virtual environment activated:

```powershell
python -m src.evaluate --episodes 100 --agent-seed 42 --env-seed 42
```

The evaluation uses the slippery 4x4 FrozenLake environment.
Each episode ends when `terminated` or `truncated` is true.

The agent seed controls action selection. The environment seed is incremented
for each episode: 42, 43, ..., 141 in this example.
Repeating the command recreates the agent and reproduces the results
with the same software versions.

### Initial results

| Metric | Value |
|---|---:|
| Episodes | 100 |
| Agent seed | 42 |
| Initial environment seed | 42 |
| Success rate | 4.00% |
| Average reward | 0.0400 |
| Average episode length | 7.95 actions |

These results describe one seeded experiment.
Success rate is the main performance metric; short episodes can also
result from falling into a hole.

## Tests

Run from the project root:

```powershell
python -m pytest
```

## Q-Learning training

The Q-Learning agent stores action-value estimates in a Q-table.
For 4x4 FrozenLake, the table contains 16 states and 4 actions.

Run training from the project root, with the virtual environment activated:

```powershell
python -m src.train --episodes 10000 --alpha 0.1 --gamma 0.99 --epsilon 0.3 --agent-seed 42 --env-seed 42 --output-dir results/q_learning
```

### Parameters

| Option | Meaning | Default |
|---|---|---:|
| `--episodes` | Number of training episodes | 10000 |
| `--alpha` | Learning rate | 0.1 |
| `--gamma` | Discount factor for future rewards | 0.99 |
| `--epsilon` | Exploration probability, constant during training | 0.3 |
| `--agent-seed` | Seed for action selection | 42 |
| `--env-seed` | Initial environment seed, incremented per episode | 42 |
| `--output-dir` | Destination folder | `results/q_learning` |

Training uses the slippery 4x4 FrozenLake environment.

Terminal transitions use only the immediate reward as the update target.
Time-limit truncations stop the episode while retaining the next-state
value estimate in the update.

### Saved artifacts

Training creates:
- `q_table.npy`: the learned Q-table, reloadable with NumPy.
- `parameters.json`: environment settings, training parameters, seeds,
  and Gymnasium and NumPy versions.

Running training again with the same output directory replaces these files.
Generated artifacts in `results/q_learning/` are ignored by Git.

Training reproducibility requires the same parameters, seeds, and software
versions. Performance evaluation will be documented separately.