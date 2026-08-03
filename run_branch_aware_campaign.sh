#!/usr/bin/env bash
# Complete resumable branch-aware publication campaign.
#
# Usage:
#   bash run_branch_aware_campaign.sh [RESULT_ROOT] [JOBS]
#
# Default RESULT_ROOT: results/normalized_metric_final
# Default JOBS: 2 (use 3-4 only on a machine with sufficient RAM)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RESULT_ROOT="${1:-results/normalized_metric_final}"
JOBS="${2:-${FPDE_REPRO_JOBS:-2}}"
export FPDE_REPRO_JOBS="$JOBS"
mkdir -p "$RESULT_ROOT/main" "$RESULT_ROOT/figures" \
         "$RESULT_ROOT/additive_gaussian" "$RESULT_ROOT/yu_weak"

# Record enough environment metadata to interpret wall-clock timings.
uname -a > "$RESULT_ROOT/system.txt" 2>&1 || true
if command -v system_profiler >/dev/null 2>&1; then
  system_profiler SPHardwareDataType > "$RESULT_ROOT/hardware.txt" 2>&1 || true
elif command -v lscpu >/dev/null 2>&1; then
  lscpu > "$RESULT_ROOT/hardware.txt" 2>&1 || true
fi

printf '\n[1/4] Running the complete test suite...\n'
python3 -u -m pytest -q | tee "$RESULT_ROOT/pytest.log"

printf '\n[2/4] Regenerating all internal manuscript experiments...\n'
bash reproduce/run_all.sh \
  --jobs "$JOBS" \
  --outdir "$RESULT_ROOT/main" \
  --figdir "$RESULT_ROOT/figures" \
  2>&1 | tee "$RESULT_ROOT/run_all.log"

printf '\n[3/4] Regenerating the additive-Gaussian robustness experiment...\n'
python3 -u scripts/run_alternative_robustness.py \
  --conditions additive_gaussian \
  --noise 10 \
  --benchmarks paper_FADE_tsfade_fft synthetic_fractional_burgers \
  --methods "Weak-Pareto" "Strong Pareto" \
  --seeds 0 1 2 3 4 \
  --jobs "$JOBS" \
  --outdir "$RESULT_ROOT/additive_gaussian" \
  2>&1 | tee "$RESULT_ROOT/additive_gaussian/run.log"

printf '\n[4/4] Regenerating only the affected Weak-Pareto rows of the Yu comparison...\n'
python3 -u scripts/run_yu2025_comparison.py \
  --output-dir "$RESULT_ROOT/yu_weak" \
  --datasets paper_FADE_tsfade_fft \
  --methods weak_pareto \
  --noise-levels 0 1 5 \
  --seeds 0 1 2 3 4 \
  --weak-profile paper \
  --weak-test-budget paper \
  --yu-profile paper \
  --yu-protocol matched \
  --yu-order-mode bank \
  --device cpu \
  --quiet \
  2>&1 | tee "$RESULT_ROOT/yu_weak/run.log"

find "$RESULT_ROOT" -type f | sort > "$RESULT_ROOT/file_manifest.txt"
python3 -m pip freeze > "$RESULT_ROOT/pip_freeze.txt"
python3 --version > "$RESULT_ROOT/python_version.txt" 2>&1

printf '\nComplete. Results are under: %s\n' "$RESULT_ROOT"
printf 'Create the upload archive with:\n  zip -r normalized_metric_final_results.zip "%s"\n' "$RESULT_ROOT"
