# Distributed Safety-Critical Control for Multi-Agent Systems

**Status: Research Prototype — Active Development**

Research software for optimization-based control of multi-robot teams. Each agent computes its control input by solving its own quadratic program (QP) with safety, connectivity and task constraints. Coupling between agents is resolved by a fast primal–dual process driven by neighbor information. The repository contains the controller, a discrete-time simulator, a centralized baseline, and tooling that independently checks the feasibility of every QP solved during a simulation.

**Tech stack:** Python · NumPy · SciPy (HiGHS LP solver) · OSQP · CVXPY · Matplotlib · PyYAML · pytest

## Research problem

$N$ agents move in the plane with single-integrator dynamics $\dot{x}_i = u_i$. The team must reach an elliptical target region, encoded with a control Lyapunov function (CLF), while:

- avoiding collisions with each other and with circular obstacles, encoded with control barrier functions (CBFs);
- keeping the communication graph connected, measured by its algebraic connectivity $\lambda_2$;
- respecting actuator limits.

Collision avoidance, connectivity and the team objective couple the agents' decisions. The research question is how to enforce these coupled constraints when each agent computes its own input, and when the per-agent problems remain feasible.

## Method

1. **Local QP per agent.** At each control step, agent $i$ solves a two-variable QP: minimize $\tfrac{1}{2}\lVert u_i\rVert^2$ subject to its collision, obstacle, connectivity, CLF and actuator constraints.
2. **Mismatch variables.** Coupled constraints are split between agents using local mismatch variables that shift each agent's constraint bounds.
3. **Fast primal–dual dynamics.** Between control steps, each agent updates its mismatch and dual variables on a faster time scale. Neighbor terms are weighted by a smooth communication weight that is zero beyond the communication range $d_c$.
4. **Centralized baseline.** One joint QP over all inputs and mismatch variables enforces the same constraints for comparison.

**Local and central computation in the current simulator.** All agents run in a single process.

| Computed per agent (local) | Computed centrally (global) |
|---|---|
| Local QP construction and solution | Algebraic connectivity $\lambda_2$ and its eigenvector (Fiedler vector), from all agent positions at every step |
| Mismatch and dual updates, with neighbor coupling limited to range $d_c$ | |

The connectivity constraint therefore uses global information, and a distributed estimate of $\lambda_2$ is not implemented.

## Implementation and engineering

- **QP construction and solving.** Each local QP is assembled from labeled affine constraints, so every row can be traced to its source. Rows are rescaled for numerical conditioning. QPs are solved with OSQP, and solver status, iteration counts, residuals and timing are recorded. (`controller/local_qp.py`)
- **Two-time-scale simulation.** The simulator runs a synchronous loop: solve all local QPs, update the fast primal–dual states, then advance the agents. It includes a numerical guard against diverging states. (`simulation/simulator.py`, `controller/fast_dynamics.py`)
- **Independent verification.** For every QP, a separate linear program (SciPy/HiGHS) computes a feasibility margin. Each solve is then classified, for example as solved, infeasible, numerically failed, or *inconsistent* (the solver reports success but the LP finds the QP infeasible). The recorder only observes and never alters the applied control. (`diagnostics/`)
- **Explicit failure semantics.** Runs stop on a QP failure instead of applying a fallback input. Each run ends with a typed termination reason, and reports cover only the interval that was actually simulated.
- **Reproducibility.** Each run writes to a new, collision-safe output directory. The configuration, command line, git commit and initial conditions are recorded, and initial positions are generated from a seed and validated.
- **Testing.** A pytest suite covers constraint construction, initial-condition generation, termination reporting, output handling and diagnostics. The development branches add tests for solver equivalence, multi-rate integration and batch execution.

## Branches

The branches form a linear history; each includes everything from the branch above it.

| Branch | Adds | Status |
|---|---|---|
| `main` | Distributed local-QP controller, fast dynamics, centralized baseline, simulator, per-QP feasibility diagnostics | Baseline |
| `feature/fast-substeps` | Multi-rate integration (configurable fast substeps per control step); direct OSQP interface replacing CVXPY in the local-QP path, with tests requiring identical results | Feature complete |
| `feature/parallel-scenarios` | Parallel batch runner: one process per scenario, YAML scenario specs with parameter overrides, CPU-allocation-aware worker count, timeouts, summary tables | Feature complete |
| `research/feasibility-study` | Feasibility certificates for recorded runs, static feasibility classification at the centralized and local levels, configurable CLF coupling, 48-scenario study tooling | In progress; results not finalized |

## Quick start

Tested with Python 3.13 on macOS. Run commands from the repository root.

```bash
git clone https://github.com/shinyuc/distributed_multi_agent_system.git
cd distributed_multi_agent_system
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run a short 5-agent simulation (500 control steps) with per-QP feasibility checks, then plot the diagnostics:

```bash
python -m diagnostics.run_feasibility --config configs/default_5_agents.yaml --max-steps 500 --out outputs/quickstart
python -m diagnostics.plot_feasibility outputs/quickstart
```

`outputs/quickstart/` then contains the run report, trajectories, per-QP logs, a run summary and `feasibility_diagnostics.png`. To run again into the same directory, add `--overwrite`; earlier results are moved aside, not deleted.

## Tests

Run the functional test suite:

```bash
pip install pytest
python -m pytest --ignore=tests/test_constraints.py --ignore=tests/test_controller_interfaces.py --ignore=tests/test_simulator.py
```

The three excluded modules test the legacy `src/dmasim` scaffold and currently fail at collection with `ModuleNotFoundError: No module named 'dmasim.env'`. Because of this, a plain `python -m pytest` stops during collection with 3 errors and runs no tests. To run everything that can be collected and still see those errors, use `python -m pytest --continue-on-collection-errors`.

## Further usage

**Full simulation.** `python main.py --config configs/default_5_agents.yaml > run.log` simulates the full configured horizon (200,000 steps) and writes trajectories, metrics and plots to a new directory under `outputs/`. Full runs are long and produce verbose logs.

**Reproduce a known failure.** `python main.py --config configs/repro/step0_failure_2026-03-25.yaml` reproduces a recorded local-QP infeasibility. It stops at step 0 by design.

**Batch execution** (on `feature/parallel-scenarios`). First switch to the branch and install its requirements:

```bash
git checkout feature/parallel-scenarios && pip install -r requirements.txt
```

Next, create a scenario spec and save it as `example_batch.yaml`. Each scenario overrides parameters of a base configuration; relative `base_config` paths are resolved against the spec's directory, then the repository root.

```yaml
name: substeps_sweep
base_config: configs/default_5_agents.yaml
overrides: {simulation.steps: 2000}
scenarios:
  - id: m1
  - id: m10
    overrides: {fast_dynamics.substeps: 10}
```

Then run the batch:

```bash
python -m diagnostics.run_batch --spec example_batch.yaml --workers 2
```

Results, including `summary.csv` and per-scenario outputs, are written to a new directory under `outputs/batches/`.

## Repository structure

```
main.py, config.py   Simulation entry point; typed, validated YAML configuration
controller/          Local QP, fast dynamics, distributed and centralized controllers
simulation/          Simulation loop and run reports
diagnostics/         Feasibility LP, per-QP recorder, plots, run summaries
models/, utils/      Data structures, initial-condition generation, output directories
configs/             Experiment configurations
tests/               pytest suite
src/dmasim/          Legacy scaffold (not used; see limitations)
```

## Known limitations

- **Connectivity information is global.** $\lambda_2$ and the Fiedler vector are computed centrally, and all agents run in one process (see [Method](#method)).
- **Legacy code.** `src/dmasim` is an incomplete earlier scaffold. Because of it, `pip install .` does not install the working code, and three test modules fail to import (see [Tests](#tests)). Use `requirements.txt` as shown above.
- **Configurations.** `default_5_agents.yaml`, `distributed_10_agents.yaml` and the `repro/` configuration run with the current code. Some older configuration files do not.
- **Dependencies.** Versions are not pinned. The development branches require OSQP ≥ 1.0, which is not yet enforced.
- **Testing scope.** Tested on macOS with Python 3.13 only.

## Ongoing work

Active research, on `research/feasibility-study`, studies when the agents' local QPs remain feasible along closed-loop trajectories and how to verify this numerically. It uses feasibility certificates for recorded runs and static feasibility checks across a set of designed scenarios. Results from this study are not yet finalized.

## Publication

Citation information will be added.

## Author

Shiyu Cheng
