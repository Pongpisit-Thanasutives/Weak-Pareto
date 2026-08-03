#!/usr/bin/env bash
# Reproduce the two-dimensional Weak-Pareto experiments.
set -euo pipefail
cd "$(dirname "$0")"

# Avoid BLAS oversubscription when seed shards run in parallel.  Use
# WEAK_PARETO_BLAS_THREADS to override the safe one-thread default.
BLAS_THREADS="${WEAK_PARETO_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export OMP_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export NUMEXPR_NUM_THREADS="$BLAS_THREADS"

DATA="${DATA:-data}"
OUT="${OUT:-results}"
PY="${PYTHON:-python3}"
NT="${NT:-90}"
NG="${NG:-80}"
JOBS="${JOBS:-5}"
SEEDS=5
NOISE="0,0.01,0.05,0.10,0.20"
CMAX=4
MAXITER=24
POPSIZE=7

check_deps() {
  "$PY" - <<'PY'
import importlib.util, sys
missing = [name for name in ("numpy", "scipy", "mpmath") if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("missing packages: " + ", ".join(missing))
PY
}

generate() {
  "$PY" generate_2d_benchmarks.py --outdir "$DATA" --nt "$NT" --ng "$NG" --which A,B
}

summarize() {
  "$PY" run_2d_experiments.py --out "$OUT" --summarize-only
}

run_sharded() {
  local experiment="$1" noise="$2" pids=() bucket seed job
  for ((job=0; job<JOBS; job++)); do
    bucket=""
    for ((seed=job; seed<SEEDS; seed+=JOBS)); do
      bucket="${bucket:+$bucket,}$seed"
    done
    [[ -z "$bucket" ]] && continue
    "$PY" run_2d_experiments.py \
      --data "$DATA" --out "$OUT" --experiment "$experiment" \
      --seed-list "$bucket" --tag "${experiment}_j${job}" --noise "$noise" \
      --cmax "$CMAX" --maxiter "$MAXITER" --popsize "$POPSIZE" &
    pids+=("$!")
  done
  local status=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}

publication() {
  rm -rf "$OUT"
  mkdir -p "$OUT"
  generate
  run_sharded A "$NOISE"
  run_sharded B "$NOISE"
  run_sharded ablation "0.05"
  summarize
  cat "$OUT/summary_2d.csv"
}

smoke() {
  [[ -f "$DATA/benchmark_A.npz" ]] || generate
  rm -rf "$OUT" && mkdir -p "$OUT"
  "$PY" run_2d_experiments.py --data "$DATA" --out "$OUT" --experiment A \
    --seed-list 0 --noise 0.05 --cmax 2 --maxiter 3 --popsize 3 --tag smoke
  summarize
}

check_deps
case "${1:-publication}" in
  publication) [[ -n "${2:-}" ]] && JOBS="$2"; publication ;;
  generate) generate ;;
  summarize) summarize ;;
  smoke) smoke ;;
  A|B|ablation)
    rm -rf "$OUT" && mkdir -p "$OUT"
    "$PY" run_2d_experiments.py --data "$DATA" --out "$OUT" --experiment "$1"
    summarize
    ;;
  *) echo "usage: $0 [publication [JOBS]|generate|summarize|smoke|A|B|ablation]" >&2; exit 2 ;;
esac
