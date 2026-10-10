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