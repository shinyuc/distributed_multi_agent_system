# Remaining 32 scenarios of the 48-scenario feasibility study

These are inputs only; nothing in this directory runs anything. Run outputs belong under `outputs/` (git-ignored).
The directory holds every scenario in `../study48/manifest.json` that is not a pilot.

| Group | Scenarios |
|---|---|
| A | A02, A05, A06, A07, A08, A10, A11, A12 |
| B | B03, B05, B06, B08, B09, B10, B11, B12 |
| C | C01, C04, C05, C06, C07, C08, C10, C12 |
| D | D01, D05, D06, D07, D09, D10, D11, D12 |

The configurations and warm starts are byte-identical copies of the generated study (`diagnostics.study48.scenarios`).
Each configuration's SHA-256 equals the value in the manifest.

| Path | Content |
|---|---|
| `configs/<ID>.yaml` | the 32 configurations |
| `warm/C04_warm.pkl`, `warm/D11_warm.pkl` | frozen warm starts, see note 1 |
| `manifest_remaining32.json` | the 32 entries of the study manifest, verbatim |
| `expected_outcomes.json` | Tier-0 predictions at x_0, see note 2 |
| `batch/remaining32_batch.yaml` | the 32-scenario batch spec, see note 3 |
| `batch/remaining32_smoke_batch.yaml` | the smoke spec, see note 4 |
| `SHA256SUMS` | checksums of all files above |

1. **Warm starts.**
   - C04: stop step 338, E = 0.0562, at the admissible radius.
   - D11: stop step 2292, E = 3.22e-3.
2. **Tier-0 predictions.** Twelve scenarios are predicted to stop at step 0 without applying a control: A05, A08,
   A10, A11, A12, B03, B05, B06, B10, B11, B12, C12. These are valid outcomes.
3. **Batch spec.** Every scenario runs `diagnostics.study48.run_pilot`:
   - the unchanged `run_feasibility` pipeline;
   - the strict feasibility guard;
   - a fast-state snapshot at every step;
   - for C04 and D11, the warm start.

   Scenarios are listed longest first so the dynamic scheduler finishes early.
4. **Smoke spec.** Five scenarios, 3 steps each:
   - A12, the legacy repro configuration;
   - C04 and D11, the warm starts;
   - C05, with M = 40;
   - A08, with N = 10 and zero obstacles.
5. **A12 is a legacy diagnostic case.** Its configuration is unchanged. Its initial state violates λ₂ ≥ η, and it
   must be excluded from any statement that all initial states satisfy the connectivity threshold.

## What each run records

All of it goes under `scenarios/<ID>/run/`.

- **Run summary.**
  - `summary.json`: termination reason and step, steps completed, QP class counts, the first non-solved QP, the
    tightest margin, and the safety and connectivity minima.
  - `run_report.json`: the run report.
  - `run_meta.json`: configuration values, versions and the git commit.
- **Per-step and per-agent data.**
  - `qp_log.csv.gz`, one row per agent and step: solver status, LP margin s*, violations, group slacks, control,
    and the connectivity, CLF and pairwise mismatch terms.
  - `step_log.csv.gz`: the graph spectrum per step.
  - `arrays.npz`: the same quantities as arrays, plus positions and λ₂.
- **Trajectories.** `trajectories.json`.
- **Fast state.** `fast_snapshots.pkl` stores the full fast state entering every step: snapshot k is the state the
  step-k QPs use.
- **Failure detail.** For every QP not verified as solved, `failures/*.json` holds:
  - all agents' positions and fast states, λ₂ and v₂;
  - the QP matrices, both raw and scaled;
  - the LP verdict and the minimal infeasible subsets;
  - the zero-mismatch counterfactual and the centralized check.

  At most 200 detail files are written per run; the summaries are not capped. With dense snapshots, any step's QP can
  also be rebuilt statically from `trajectories[k]` and `snaps[k]`.
- **Provenance.**
  - `pilot_runner.json`: strict guard, snapshot setting, config SHA-256, git commit and whether tracked files were
    modified.
  - `warm_start.json`: the warm file's SHA-256, for C04 and D11.
