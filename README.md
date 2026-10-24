# FrozenLake Reinforcement Learning

A Python project for learning and experimenting with reinforcement learning using the FrozenLake environment from Gymnasium.

## Project status

The project includes a reproducible random baseline, a tabular Q-Learning
agent, training and evaluation commands, an exploration notebook, and tests.
On the documented evaluation, Q-Learning achieved 64.6% success versus 1.4%
for the random agent. See the protocol and limitations below.

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
│   ├── train.py
│   ├── visualize.py
│   └── learning_viewer.html
├── tests/
│   ├── test_agent.py
│   ├── test_evaluate.py
│   ├── test_evaluate_extended.py
│   ├── test_train.py
│   └── test_visualize.py
├── notebooks/
│   └── exploration.ipynb
└── results/
    └── comparison.json
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
versions.

## Evaluate and compare agents

Run the training command above first to generate the ignored Q-table.
Then run from the project root with the virtual environment activated:

```powershell
python -m src.evaluate --agent compare --episodes 1000 --agent-seed 42 --env-seed 20000 --output results/comparison.json
```

To evaluate only the trained policy:

```powershell
python -m src.evaluate --agent q-learning --episodes 1000 --env-seed 20000 --q-table results/q_learning/q_table.npy
```

The original random-agent command remains available; `--agent random` is the
default. `--output` optionally saves a JSON report and replaces an existing
report at that destination. The report records seeds, metrics, library
versions, the Q-table SHA-256 fingerprint, and training parameters when the
adjacent `parameters.json` file is present.

Q-tables must contain finite real numbers with shape `(16, 4)`. Missing or
invalid tables produce an error. Evaluation selects a maximum-value action
with epsilon zero, chooses the first action on ties, and never updates the
table. Random tie-breaking used during training is intentionally replaced by
a fixed tie rule for evaluation.

### Comparison protocol and results

- Environment: slippery 4x4 FrozenLake, maximum 100 actions per episode.
- Training: 10,000 episodes, alpha 0.1, gamma 0.99, constant epsilon 0.3,
  agent seed 42, environment seeds 42 through 10041.
- Evaluation: 1,000 episodes per agent, environment seeds 20000 through 20999,
  with random-agent seed 42. Each agent starts with a freshly reset environment.
- Evaluation seeds do not overlap with training seeds. Both agents use the
  same list of environment seeds; different actions can still produce different
  trajectories and episode lengths.
- Software: Python 3.14.8, Gymnasium 1.4.0, NumPy 2.5.4; see requirements.txt.

| Agent | Wins / episodes | Success rate | Average reward | Average episode length |
|---|---:|---:|---:|---:|
| Random | 14 / 1000 | 1.40% | 0.0140 | 7.723 actions |
| Q-Learning | 646 / 1000 | 64.60% | 0.6460 | 43.782 actions |

The trained policy improves success by **63.2 percentage points** in this
experiment. Longer episodes do not alone imply better play: random episodes
often end quickly in holes, while some trained-policy episodes reach the
100-action limit. Average length includes wins, losses, and truncated episodes.

The earlier random baseline of 4% used only 100 episodes with different seeds;
the table above reruns both agents under a common evaluation protocol.

### Limitations

These results evaluate one trained table on one fixed map and one evaluation
seed range. They do not establish performance across independent training
runs, unseen maps, or different library versions. No confidence intervals or
hyperparameter search are included. Constant exploration and learning rates
were chosen for a simple educational implementation; optimality is not claimed.

The report is versioned, while generated training artifacts remain local and
can be recreated with the documented training command.

## Watch the agent learn

Generate an interactive offline viewer from the project root:

```powershell
python -m src.visualize
```

Then double-click `results/learning/index.html` in Windows File Explorer to
open it in your browser. No web server, internet connection, extra package,
or notebook extension is needed to view the exported page. The training
command prints progress while generating the replay data.

The page includes:
- Seven training stages: 0, 100, 500, 1000, 2500, 5000, and 10000 episodes.
- A clickable success-rate curve and checkpoint selector.
- Animated trajectories, play/pause, single step, timeline, and playback speed.
- A tour that plays the first recorded attempt at each checkpoint.
- The first 12 evaluation attempts with success, hole, and time-limit outcomes.
- A switch between the learned policy (epsilon zero) and a replay with the
  training exploration probability (epsilon 0.3 by default).
- Q-values for the agent's current cell and the requested action at each step.

These trajectories are **replays from checkpoint policies**, not recordings
of the historical training episodes. Each checkpoint is evaluated over 200
episodes by default, using the same environment seeds 20000 through 20199.
The first 12 are displayed without selecting particular outcomes. Exploration
replays use the same seed range but do not contribute to the success-rate curve.
All replays use separate agents and environments, so capturing checkpoints
does not change the training RNG state or final Q-table.

In this viewer, greedy ties are broken randomly with a fixed seed, matching
the agent's training tie rule. This makes the zero-table policy act randomly.
The existing comparison command retains its documented first-maximum tie rule.
The final viewer score of 60% is measured on 200 episodes, rather than the
1,000 episodes used for the 64.6% comparison report.

The default run produced these success rates:

| Training episodes | Success on 200 evaluation episodes |
|---:|---:|
| 0 | 1.5% |
| 100 | 1.5% |
| 500 | 32.0% |
| 1000 | 57.0% |
| 2500 | 70.0% |
| 5000 | 56.0% |
| 10000 | 60.0% |

Learning is not monotonic. Slips remain stochastic, and constant learning
and exploration rates can change the policy after it has improved. This curve
illustrates a single training run; it is not an independent comparison across
multiple runs and is not used here to select hyperparameters or a checkpoint.

To customize generation:

```powershell
python -m src.visualize --episodes 10000 --checkpoints 100 500 1000 2500 5000 --eval-episodes 200 --replays 12 --alpha 0.1 --gamma 0.99 --epsilon 0.3 --agent-seed 42 --env-seed 42 --eval-seed 20000 --output-dir results/learning
```

Training and evaluation seed ranges must not overlap. The generator validates
this before training; use a higher `--eval-seed` for a longer training run.
The initial and final checkpoints are always included. Each run replaces
matching files in its output directory. The folder contains `history.json`,
reloadable `q_table_XXXXXX.npy` checkpoints, and the self-contained `index.html`.
Generated files in `results/learning/` are ignored by Git. The original
`results/q_learning/` artifacts are left untouched.

## Development workflow

Work is organized through GitHub Issues, feature branches, clear commits,
tests, and Pull Requests reviewed before merging into `main`.
The exploration notebook explains the map, API, termination, truncation,
and reproducibility experiments before introducing learning.
