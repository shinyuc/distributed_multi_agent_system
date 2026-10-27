#!/usr/bin/env bash
# Server preparation for the 16-scenario pilot (study48), with all inputs taken from the Git repository.
# Does NOT run the pilot batch.
#
#   EXPECT=<commit sha> bash experiments/feasibility_study/pilot16/SERVER_STEPS.sh
#
# EXPECT    commit the research branch must be at after the update (required; given with the instructions,
#           since a committed file cannot contain its own commit hash).
# REPO_DIR  repository checkout (default /home/cytrus/Documents/Shiyu/distributed_multi_agent_system). Only branch
#           research/feasibility-study is touched (fast-forward only); main is not modified; the script aborts if
#           tracked files are modified.
# RUN_ROOT  where logs and smoke outputs go (default $REPO_DIR/outputs/feasibility_study/pilot16, git-ignored).
# PY        python command of the existing environment (default: conda run --no-capture-output -n dmasim-feas python).
set -euo pipefail

: "${EXPECT:?set EXPECT to the expected commit of research/feasibility-study}"
BRANCH=research/feasibility-study
REPO_DIR=${REPO_DIR:-/home/cytrus/Documents/Shiyu/distributed_multi_agent_system}
RUN_ROOT=${RUN_ROOT:-$REPO_DIR/outputs/feasibility_study/pilot16}
PY=${PY:-"conda run --no-capture-output -n dmasim-feas python"}
PILOT_REL=experiments/feasibility_study/pilot16
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_ROOT"
LOG="$RUN_ROOT/server_prep_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

echo "== 1. repository ($REPO_DIR): update $BRANCH only"
cd "$REPO_DIR"
[ "$(git rev-parse --show-toplevel)" = "$(pwd -P)" ] || { echo "ABORT: $REPO_DIR is not the top of a git checkout"; exit 1; }
echo "origin: $(git remote get-url origin)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ABORT: tracked files are modified in $REPO_DIR"; git status --short --untracked-files=no; exit 1
fi
echo "current branch before: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse HEAD)"
git fetch origin "$BRANCH"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout "$BRANCH"
    git merge --ff-only "origin/$BRANCH"
else
    git checkout -b "$BRANCH" --track "origin/$BRANCH"
fi
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD"
[ "$HEAD" = "$EXPECT" ] || { echo "ABORT: HEAD differs from the expected commit $EXPECT"; exit 1; }

STUDY_DIR="$REPO_DIR/$PILOT_REL"
echo "== 2. pilot inputs ($STUDY_DIR)"
( cd "$STUDY_DIR" && sha256sum -c --strict SHA256SUMS )
test "$(ls "$STUDY_DIR"/configs/*.yaml | wc -l)" -eq 16
test -f "$STUDY_DIR/warm/D03_warm.pkl"

echo "== 3. environment and tests"
$PY -c "import sys, numpy, scipy, cvxpy, osqp; print(sys.executable, sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'cvxpy', cvxpy.__version__, 'osqp', osqp.__version__)"
$PY -c "import os; from diagnostics.run_batch import detect_cpu_allocation; print(detect_cpu_allocation(), 'cpu_count', os.cpu_count())"
# src/dmasim/env/ is matched by the .gitignore rule "env/" and has never been committed (pre-existing; unrelated to
# the pilot code paths). On a checkout without it, three legacy test modules cannot be imported; skip only those.
IGNORE=()
if [ ! -d src/dmasim/env ]; then
    echo "NOTICE: src/dmasim/env missing in this checkout; skipping tests/test_constraints.py, test_controller_interfaces.py, test_simulator.py"
    IGNORE=(--ignore=tests/test_constraints.py --ignore=tests/test_controller_interfaces.py --ignore=tests/test_simulator.py)
fi
$PY -m pytest -q -p no:cacheprovider ${IGNORE[@]+"${IGNORE[@]}"}

echo "== 4. smoke batch (3 scenarios x 3 steps)"
SMOKE="$RUN_ROOT/smoke_$STAMP"
mkdir -p "$SMOKE"
$PY -m diagnostics.run_batch --spec "$STUDY_DIR/pilot/server_smoke_batch.yaml" --workers 3 --out "$SMOKE"
$PY -m diagnostics.study48.verify_smoke --bundle "$STUDY_DIR" --batch "$SMOKE" --expect-commit "$EXPECT"

echo "== 5. pilot command (NOT run):"
echo "cd $REPO_DIR && $PY -m diagnostics.run_batch --spec $PILOT_REL/pilot/pilot_batch.yaml --workers 16 --timeout 7200 --out $RUN_ROOT/pilot16_<timestamp>"
echo "log: $LOG"
