# Distributed Multi-Agent Simulation (Research Scaffold)

This project provides a modular simulation scaffold for distributed control of multi-agent systems.

## Scope
- 2D single-integrator dynamics: `x_i_dot = u_i`
- Circular obstacles and target region
- Safety constraints (inter-agent, obstacle, optional connectivity)
- Distributed and centralized QP controller entry points
- Discrete-time simulation, plotting, and method comparison

## Important
Controller implementations are intentionally structured as equation placeholders:
- `src/dmasim/controllers/distributed_qp.py`
- `src/dmasim/controllers/centralized_qp.py`
- `src/dmasim/controllers/constraints.py`
- `src/dmasim/controllers/objectives.py`

Fill these with your paper's exact equations and constants.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m dmasim.experiments.run_simulation --config configs/distributed.yaml
```

## Compare methods
```bash
python -m dmasim.experiments.compare_methods \
  --distributed-config configs/distributed.yaml \
  --centralized-config configs/centralized.yaml
```
