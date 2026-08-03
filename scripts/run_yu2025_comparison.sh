#!/usr/bin/env bash
# One-command comparison of Weak-Pareto with the adapted neural fractional-discovery framework of Yu et al. (2025).
#
# CPU:
#   FPDE_DEVICE=cpu FPDE_QUICK=1 bash scripts/run_yu2025_comparison.sh
# GPU (CUDA):
#   FPDE_DEVICE=cuda bash scripts/run_yu2025_comparison.sh
#
# The full default run uses the closest FADE benchmark; the spectral and finite-terminal operator realisations differ and are disclosed in the manuscript.
# Override any array-valued setting with a space-separated environment variable,
# e.g. FPDE_NOISE="0 1 5 25" FPDE_SEEDS="0 1 2 3 4".

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUICK="${FPDE_QUICK:-0}"
DEVICE="${FPDE_DEVICE:-auto}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_ROOT="${RESULT_ROOT:-results/yu2025_comparison_${TIMESTAMP}}"
DATASETS=( ${FPDE_DATASETS:-paper_FADE_tsfade_fft} )
METHODS=( ${FPDE_METHODS:-weak_pareto yu2025_full yu2025_optimizer_only} )
YU_PROTOCOL="${FPDE_YU_PROTOCOL:-matched}"
YU_ORDER_MODE="${FPDE_YU_ORDER_MODE:-bank}"
YU_DTYPE="${FPDE_YU_DTYPE:-float32}"

if [[ "$QUICK" == "1" ]]; then
  WEAK_PROFILE="notebook"
  WEAK_TEST_BUDGET="smoke"
  YU_PROFILE="smoke"
  NOISE=( ${FPDE_NOISE:-0} )
  SEEDS=( ${FPDE_SEEDS:-0} )
  WEAK_RUNTIME=( --weak-maxiter "${FPDE_WEAK_MAXITER:-0}" --weak-popsize "${FPDE_WEAK_POPSIZE:-2}" )
else
  WEAK_PROFILE="paper"
  WEAK_TEST_BUDGET="paper"
  YU_PROFILE="paper"
  NOISE=( ${FPDE_NOISE:-0 1 5} )
  SEEDS=( ${FPDE_SEEDS:-0 1 2 3 4} )
  WEAK_RUNTIME=()
fi

YU_OVERRIDES=()
if [[ -n "${FPDE_YU_EPOCHS:-}" ]]; then
  YU_OVERRIDES+=( --yu-epochs "$FPDE_YU_EPOCHS" )
fi
if [[ -n "${FPDE_YU_DE_MAXITER:-}" ]]; then
  YU_OVERRIDES+=( --yu-de-maxiter "$FPDE_YU_DE_MAXITER" )
fi
if [[ -n "${FPDE_YU_DE_POPSIZE:-}" ]]; then
  YU_OVERRIDES+=( --yu-de-popsize "$FPDE_YU_DE_POPSIZE" )
fi
if [[ -n "${FPDE_YU_QUADRATURE_NODES:-}" ]]; then
  YU_OVERRIDES+=( --yu-quadrature-nodes "$FPDE_YU_QUADRATURE_NODES" )
fi

mkdir -p "$RESULT_ROOT"

printf '[1/3] Running unit tests\n'
pytest -q | tee "$RESULT_ROOT/pytest.log"

printf '[2/3] Running Weak-Pareto versus Yu et al. comparison\n'
python scripts/run_yu2025_comparison.py \
  --output-dir "$RESULT_ROOT/comparison" \
  --datasets "${DATASETS[@]}" \
  --methods "${METHODS[@]}" \
  --noise-levels "${NOISE[@]}" \
  --seeds "${SEEDS[@]}" \
  --weak-profile "$WEAK_PROFILE" \
  --weak-test-budget "$WEAK_TEST_BUDGET" \
  --yu-profile "$YU_PROFILE" \
  --yu-protocol "$YU_PROTOCOL" \
  --yu-order-mode "$YU_ORDER_MODE" \
  --yu-dtype "$YU_DTYPE" \
  --device "$DEVICE" \
  ${WEAK_RUNTIME[@]+"${WEAK_RUNTIME[@]}"} \
  ${YU_OVERRIDES[@]+"${YU_OVERRIDES[@]}"}

printf '[3/3] Writing run metadata\n'
cat > "$RESULT_ROOT/RUN_METADATA.txt" <<EOF
result_root=$RESULT_ROOT
quick=$QUICK
device=$DEVICE
datasets=${DATASETS[*]}
methods=${METHODS[*]}
noise=${NOISE[*]}
seeds=${SEEDS[*]}
weak_profile=$WEAK_PROFILE
weak_test_budget=$WEAK_TEST_BUDGET
yu_profile=$YU_PROFILE
yu_protocol=$YU_PROTOCOL
yu_order_mode=$YU_ORDER_MODE
yu_dtype=$YU_DTYPE
EOF
cp "$RESULT_ROOT/comparison/SUMMARY.md" "$RESULT_ROOT/COMBINED_SUMMARY.md"

printf 'Done. Results: %s\n' "$RESULT_ROOT"
printf 'Summary: %s\n' "$RESULT_ROOT/COMBINED_SUMMARY.md"
