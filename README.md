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
└── src/
    └── __init__.py
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