# Validation of Resource Allocation Policies via Reachability

Implementation for the paper **"Feasibility of Reachability to Validate Resource Allocation Policies Under Temporal Logic Specifications"** (AA228V/CS238V Final Project, Stanford University).

This framework validates closed-loop scheduling policies under stochastic uncertainty using two complementary methods: **Monte Carlo simulation** (statistical failure estimates) and **SAT-based bounded reachability** (worst-case counterexamples via PySAT).

## Overview

- **Domain**: Task coordination with limited resources, precedence constraints, and stochastic delays (duration noise, delay events).
- **Specifications**: Four LTL-style properties—resource overload, precedence violation, task completion, no abandonment.
- **Policies**: Greedy, Conservative, and Priority-based deterministic schedulers.
- **Validation**: Monte Carlo estimates violation probabilities; SAT-based bounded model checking finds concrete violating disturbance sequences within a finite horizon.

## Setup

```bash
pip install -r requirements.txt
```

Requirements: Python 3.10+, `python-sat`, `matplotlib`.

## Usage

Run the full benchmark (generates environments, runs MC + SAT, prints results, saves figure):

```bash
python main.py
```

### Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--task-sizes` | `4,6,8,10,12` | Comma-separated task counts |
| `--env-seeds` | `5` | Random environments per task size |
| `--mc-trials` | `600` | Monte Carlo trials per policy/environment |
| `--sat-timeout-sec` | `30.0` | SAT timeout (seconds) per policy/environment |

### Example

```bash
python main.py --task-sizes 4,6,8 --env-seeds 8 --mc-trials 500 --sat-timeout-sec 15
```

Output: console summary of violation rates and SAT outcomes, plus `results.png` (MC completion failure vs task count, SAT status heatmap).

## Reproducing Paper Results

To reproduce the benchmark in the paper:

```bash
python main.py --task-sizes 4,6,8,10,12 --env-seeds 5 --mc-trials 600 --sat-timeout-sec 30
```

The figure `results.png` matches the two-panel plot in the paper. Tables 1 and 2 are derived from this run.

## Project Structure

```
.
├── main.py                 # Entry point: benchmark driver
├── requirements.txt
├── results.png             # Generated summary figure
└── src/
    ├── model/              # Core domain
    │   ├── env.py          # Environment, tasks, resources, precedence, generator
    │   ├── state.py        # State representation, initial state
    │   ├── policy.py       # Greedy, Conservative, Priority policies
    │   └── scheduler.py    # Stochastic/deterministic transition
    ├── validation/         # Validation methods
    │   ├── simulation.py   # Monte Carlo, property monitors
    │   └── sat_reachability.py  # SAT-based bounded reachability (PySAT)
    └── analysis/           # Benchmark orchestration
        ├── analysis.py     # run_benchmark, aggregation
        └── visualization.py  # plot_results
```

## Key Components

- **`src/model/`**: Environment generator scales resources, precedence density, and horizon with task count. Policies enforce precedence and resource capacity by construction.
- **`src/validation/simulation.py`**: Trajectory simulation under disturbance model; monitors for the four LTL-style properties.
- **`src/validation/sat_reachability.py`**: Builds bounded transition graph, encodes path-finding as CNF, uses Glucose3 to find violating trajectories.
- **`src/analysis/`**: Runs MC + SAT across task sizes and seeds, aggregates results, produces the summary figure.

## Citation

If you use this code, please cite the course and the associated report:

```bibtex
@misc{mehta2024reachability,
  title={Feasibility of Reachability to Validate Resource Allocation Policies Under Temporal Logic Specifications},
  author={Mehta, Yash and Mendoza-Perez, Adrian},
  year={2024},
  note={AA228V/CS238V Final Project, Stanford University}
}
```
