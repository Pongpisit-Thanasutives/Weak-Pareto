#!/usr/bin/env bash
# Generate the reproducible benchmark outputs for the paper.
#
# Proposed method:
#   weak_pareto = weak candidate library + best-subset Pareto-DE
#
# Baselines:
#   vanilla_pareto        = vanilla/strong library + best-subset Pareto-DE
#   weak_grid_stridge     = weak library + alpha-grid STRidge
#   weak_fixed_stability  = fixed weak grid library + stability-selected STRidge
#
# Use FPDE_QUICK=1 for a short sanity check.  Omit it for the paper profile.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUICK="${FPDE_QUICK:-0}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_ROOT="${RESULT_ROOT:-results/publication_${TIMESTAMP}}"

if [[ "$QUICK" == "1" ]]; then
  PROFILE="notebook"
  NOISE=( ${FPDE_NOISE:-0 0.1} )
  SEEDS=( ${FPDE_SEEDS:-0} )
  DATASETS=( ${FPDE_DATASETS:-synthetic_space_fractional_RD} )
  MAXITER_FLAGS=( --maxiter "${FPDE_MAXITER:-0}" )
  POPSIZE_FLAGS=( --popsize "${FPDE_POPSIZE:-2}" )
  WEAK_TEST_BUDGET="${FPDE_WEAK_TEST_BUDGET:-smoke}"
  STABILITY_SPLITS="${FPDE_STABILITY_SPLITS:-1}"
  STABILITY_WIDTH_SCALES=( ${FPDE_STABILITY_WIDTH_SCALES:-1.0} )
else
  PROFILE="paper"
  # The final paper can override these.  Defaults emphasize the low-noise range
  # where structure recovery, order errors, and coefficient errors are meaningful.
  NOISE=( ${FPDE_NOISE:-0 0.1 0.25 0.5 1 2} )
  SEEDS=( ${FPDE_SEEDS:-0 1 2 3 4} )
  DATASETS=( ${FPDE_DATASETS:-} )
  MAXITER_FLAGS=()
  POPSIZE_FLAGS=()
  WEAK_TEST_BUDGET="${FPDE_WEAK_TEST_BUDGET:-paper}"
  STABILITY_SPLITS="${FPDE_STABILITY_SPLITS:-5}"
  STABILITY_WIDTH_SCALES=( ${FPDE_STABILITY_WIDTH_SCALES:-0.85 1.0 1.2} )
fi

mkdir -p "$RESULT_ROOT"

printf '[1/5] Running unit tests\n'
pytest -q | tee "$RESULT_ROOT/pytest.log"

printf '[2/5] Exporting canonical dataset configs\n'
python scripts/export_canonical_configs.py
cp configs/canonical_dataset_configs.json "$RESULT_ROOT/canonical_dataset_configs.json"

DATASET_FLAGS=()
if [[ ${#DATASETS[@]} -gt 0 ]]; then
  DATASET_FLAGS=( --datasets "${DATASETS[@]}" )
fi

LIBRARY_FLAGS=( --cmax "${FPDE_CMAX:-4}" )
if [[ -n "${FPDE_P_VALUES:-}" ]]; then
  LIBRARY_FLAGS+=( --p-values ${FPDE_P_VALUES} )
fi

printf '[3/5] Running paper comparison: proposed weak_pareto vs baselines\n'
python scripts/run_all_methods.py \
  --profile "$PROFILE" \
  --methods weak_pareto vanilla_pareto weak_grid_stridge weak_fixed_stability \
  --noise-levels "${NOISE[@]}" \
  --seeds "${SEEDS[@]}" \
  --weak-test-budget "$WEAK_TEST_BUDGET" \
  --stability-splits "$STABILITY_SPLITS" \
  --stability-width-scales "${STABILITY_WIDTH_SCALES[@]}" \
  ${DATASET_FLAGS[@]+"${DATASET_FLAGS[@]}"} \
  ${MAXITER_FLAGS[@]+"${MAXITER_FLAGS[@]}"} \
  ${POPSIZE_FLAGS[@]+"${POPSIZE_FLAGS[@]}"} \
  ${LIBRARY_FLAGS[@]+"${LIBRARY_FLAGS[@]}"} \
  --quiet \
  --output-dir "$RESULT_ROOT/method_comparison"

printf '[4/5] Summarizing and checking zero-noise proposed-method recovery\n'
python scripts/summarize_publication_results.py \
  --inputs "$RESULT_ROOT/method_comparison" \
  --output-md "$RESULT_ROOT/method_comparison/SUMMARY.md" \
  --require-proposed-zero-full

printf '[5/5] Writing run metadata\n'
cp "$RESULT_ROOT/method_comparison/SUMMARY.md" "$RESULT_ROOT/COMBINED_SUMMARY.md"
cat > "$RESULT_ROOT/RUN_METADATA.txt" <<EOF
result_root=$RESULT_ROOT
quick=$QUICK
profile=$PROFILE
noise=${NOISE[*]}
seeds=${SEEDS[*]}
datasets=${DATASETS[*]:-all packaged datasets}
weak_test_budget=$WEAK_TEST_BUDGET
stability_splits=$STABILITY_SPLITS
stability_width_scales=${STABILITY_WIDTH_SCALES[*]}
proposed_method=weak_pareto (weak library + best-subset Pareto-DE)
baselines=vanilla_pareto weak_grid_stridge weak_fixed_stability
library_cmax=${FPDE_CMAX:-4}
library_override_p_values=${FPDE_P_VALUES:-canonical}
EOF

printf 'Done. Results are in: %s\n' "$RESULT_ROOT"
printf 'Summary: %s\n' "$RESULT_ROOT/COMBINED_SUMMARY.md"
