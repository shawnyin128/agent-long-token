#!/usr/bin/env bash
# Run a sequence of grid cells; commit + push artifacts after each one.
#
# Each argument is a cell spec ("qwen3:gsm8k:0:adversarial-strict" etc).
# The corresponding cell_dir is computed from the spec via Python so
# this stays in sync with agentdiet/grid/types.cell_dir().
#
# vLLM SERVER MUST BE RUNNING for the cell's model_family before invoking.
# This script does NOT start or stop vLLM.
#
# Env tweakable:
#   CALIBRATION_PREFIX  — first N questions used to calibrate voting (default 5)
#   MAX_CONCURRENCY     — concurrent question requests per condition (default 4)
#   N_QUESTIONS         — questions per cell, passed to --n (default unset = 40)
#   SKIP_PUSH           — set to 1 to skip git push after each commit
#
# Usage examples:
#   bash scripts/run_grid_cells.sh \
#     qwen3:gsm8k:1 \
#     qwen3:gsm8k:0:adversarial-strict \
#     qwen3:gsm8k:1:adversarial-strict \
#     qwen3:gsm8k:0:symmetric \
#     qwen3:gsm8k:1:symmetric
#
#   bash scripts/run_grid_cells.sh gpt-oss:gsm8k:0 gpt-oss:gsm8k:1
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "ERROR: no cell specs provided"
    echo "Usage: $0 CELL_SPEC [CELL_SPEC ...]"
    echo "  e.g.: $0 qwen3:gsm8k:1 qwen3:gsm8k:0:symmetric"
    exit 2
fi

CALIBRATION_PREFIX="${CALIBRATION_PREFIX:-5}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
SKIP_PUSH="${SKIP_PUSH:-0}"

# Optional --n flag
N_FLAG=()
if [[ -n "${N_QUESTIONS:-}" ]]; then
    N_FLAG=(--n "$N_QUESTIONS")
fi

# Resolve cell_dir from cell spec via the actual project code so
# naming stays consistent.
resolve_cell_dir () {
    local spec="$1"
    python -c "
import sys
from agentdiet.cli.grid import parse_cell_spec
from agentdiet.grid.types import cell_dir
print(cell_dir(parse_cell_spec(sys.argv[1])))
" "$spec"
}

total=$#
i=0
for spec in "$@"; do
    i=$((i + 1))
    cell_dir=$(resolve_cell_dir "$spec")
    artifact_path="artifacts/grid/$cell_dir"

    echo
    echo "============================================================"
    echo "[$i/$total] Cell: $spec"
    echo "         dir: $artifact_path"
    echo "============================================================"

    python -m agentdiet.cli.grid \
        --cell "$spec" \
        --calibration-prefix "$CALIBRATION_PREFIX" \
        --max-concurrency "$MAX_CONCURRENCY" \
        "${N_FLAG[@]}"

    if [[ ! -d "$artifact_path" ]]; then
        echo "ERROR: expected artifact directory $artifact_path not found"
        exit 1
    fi

    git add -f "$artifact_path"
    if git diff --cached --quiet; then
        echo "[$i/$total] no new changes to commit (cell already up-to-date?)"
    else
        git commit -m "[grid]: $spec cell"
        if [[ "$SKIP_PUSH" != "1" ]]; then
            git push origin HEAD || echo "WARNING: push failed; continue anyway"
        fi
    fi

    echo "[$i/$total] DONE"
done

echo
echo "============================================================"
echo "All $total cells done."
echo "============================================================"
