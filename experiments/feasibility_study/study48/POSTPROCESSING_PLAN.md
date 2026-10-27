# Automated post-processing of the 48-scenario dataset

This is a plan only; nothing here is implemented yet. It runs after all 48 scenarios have results. No new closed-loop
simulation is needed: every quantity below is computed statically from the recorded outputs.

## Inputs

- **Dataset directory.** One entry per scenario id, linking to its run directory:
  - pilots: the pilot batch;
  - D03: the corrected rerun (`D03_rerun_*`). The original D03 output is kept unchanged, and it is listed separately
    as `D03_original_debug_error`;
  - the other 32: the remaining-32 batch.
- **Static files in this directory.** `manifest.json` (all 48), plus `tier0_summary_all.json` and
  `tier0_summary_all.csv` (revision 2).
- **Configurations and warm starts.** In `../pilot16/` and `../remaining32/`.

## Stage P0: integrity and provenance (per scenario)

Each check is reported as PASS or FAIL:

- the configuration that ran is identical, as parsed YAML, to the repository file, and its SHA-256 equals the
  manifest value;
- the warm file's SHA-256 matches, for D03, C04 and D11;
- the git commit is the expected one and no tracked files were modified;
- the strict guard was on, and the snapshot setting is recorded;
- the batch status is one of completed, terminated, error or timeout.

`error` and `timeout` are flagged as data problems. They are not experimental outcomes.

## Stage P1: run outcome table (per scenario)

The table holds:

- status, steps completed out of steps configured, termination reason and step;
- the agents whose QPs failed and the failure kind:
  - `solver_reported_infeasible`;
  - `solver_numerical_failure`;
  - strict guard, meaning the solver reported success but the LP is infeasible;
- QP class counts (solved, solved_inaccurate, borderline, infeasible, numerical_failure, inconsistent, lp_error)
  and the tightest LP margin;
- minimum pairwise distance minus d₀, minimum clearance, minimum λ₂ − η, and target entry;
- the largest connectivity, CLF and pairwise mismatch terms.

## Stage P2: static classification at the terminating step

The state at termination step k consists of every agent's position and fast state. It is taken from
`failures/*.json` when present; otherwise it is rebuilt from `trajectories[k]` and `snaps[k]`. If neither is
available, which can only happen for pilot runs with sparse snapshots whose detail files hit the cap, the state is
reported as missing.

The checks, run with `diagnostics.static_feasibility.static_check(model, x_k, z_k)`:

- **Levels 1–3**, with confirmed F/I/U verdicts:
  - an F verdict carries a re-verified witness;
  - an I verdict carries a Farkas certificate with its support and φ, or a zero-row certificate with its floor;
  - U carries the reason.
- **Category at failure:** A, B, C, feasible or U.
- **Lifted Prop 2 allocation** for Level 2 F, with a rounding-error bound.
- **Augmented-LP bounded-mismatch evidence** when the augmented verdict is U.
- **Disagreement diagnosis** (`diagnostics.study48.disagreement`) whenever the componentwise Level 2, the augmented
  LP and `centralized_feasibility` disagree. The evidence covers:
  - exact rational verification of the witness;
  - the dual lower bound on ‖z‖∞;
  - coupling weights listed exactly, with tiny weights never treated as zero;
  - the HiGHS small-coefficient mechanism.

This stage is validated: it is the computation the regression tests already apply to recorded failure JSONs.

## Stage P3: numerical-consistency checks

- **Per agent at failure:** the recorder's LP verdict versus the static Level-3 verdict, and the OSQP status versus
  the LP verdict.
- **Borderline QPs** with |s*| ≤ 1e-6, and QPs in the inconsistent class.
- **Zero rows** inside the tolerance band.
- **Model identity checks** (`check_fast_dynamics`, `check_local_qp`) at sampled states. These confirm that the
  certifier rows equal the implementation.

## Stage P4: feasibility along the trajectory (sampled)

Every 10th step plus the last 50 steps before termination, using `trajectories[k]` and `snaps[k]`:

- Level 1 and 2 margins at x_k;
- Level 3 margins at the actual mismatch, and at the zero mismatch as a counterfactual;
- the time of the first Level-3 I verdict, compared with the recorded termination step.

The cost is about 20 ms per state.

## Stage P5: certificates

For completed runs and for the prefixes of terminated runs:

- `certify` under the spectral norm (primary) and the Frobenius norm (comparison): status, certified states, first
  obstruction, Γ and the minimum gap;
- ρ reported as a certified lower bound (≥ 0), an upper bound, and a status;
- `upper_bound_below_tol` means ρ is numerically zero. It is not a proof that ρ = 0, unless the bounding ray is a
  single z-only row (`structural_zero_rows`) that is exactly active;
- observed errors from the dense snapshots.

## Stage P6: synthesis

Per scenario:

- Tier-0 prediction, dynamic outcome, category at failure, and certificate outcome;
- prediction versus outcome, with mismatches flagged.

For the comparison pairs of the manifest, listing like against like:

- single-parameter pairs, with attribution;
- design-factor pairs, with side effects stated;
- broader contrasts, descriptive only.

Outputs, written outside the repository input directories:

- one JSON per scenario;
- a combined CSV and JSON;
- a Markdown report with the F/I/U and ambiguity counts.

## Reporting conventions

- F and I are numerically verified verdicts, F by witness and I by certificate. They become exact statements only
  where an exact-arithmetic check or a structural argument is given.
- U means numerical ambiguity and is never forced to F or I.
- A solver disagreement is reported with its diagnosed mechanism.
- Certificates are sufficient conditions only.
- Early termination caused by infeasibility is a valid outcome.

## Implementation (proposed, after approval)

The plan would be implemented as one module, `diagnostics/study48/postprocess.py`.

- Command line: `--dataset DIR --out DIR --workers N`, with one worker per scenario.
- It reuses `static_feasibility`, `certificate`, `disagreement` and the recorder file formats; nothing new is needed
  in the controller or the simulator.
- Unit tests would use synthetic run directories, and the recorded A, C and D3 outputs where available.
