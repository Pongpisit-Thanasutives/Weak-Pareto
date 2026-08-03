#!/usr/bin/env bash
#
# run_all.sh -- regenerate the manuscript's INTERNAL tables and figures.
#
# This driver does NOT run the Yu et al. neural fractional-discovery comparison (Table with
# label tab:yu / Appendix I); regenerate that separately with
#   scripts/run_yu2025_comparison.sh
#
# By default this runs at the full experiment budgets, i.e. the settings
# used for the non-"--fast" path in reproduce/_repro_common.py:
#   * seeds 0..4 (5 seeds)
#   * differential evolution SciPy population multiplier x generations = 7 x 24
#     (the multiplier 7 gives 7*(c+1) individuals for a support of size c)
#   * noise levels 0, 1, 5, 10, 20 (%)
# This is intentionally slow (the weak-vs-strong sweep dominates the runtime).
#
# Usage:
#   reproduce/run_all.sh [--fast] [--smoke] [-j N|--jobs N|--jobs=N]
#                        [--figdir DIR] [--outdir DIR]
#
#     (no flags)     full publication budget
#     -j, --jobs N   parallel workers for independent runs (default 2;
#                    alternatively set FPDE_REPRO_JOBS)
#     --figdir DIR   write figures into DIR (e.g. your manuscript's figures/
#                    directory, to overwrite the placeholders); default results/figures
#     --outdir DIR   write the table CSV and .tex rows into DIR; default results
#     --fast         small budget: quick, NOT publication quality
#     --smoke        tiny deterministic Burgers weak/strong self-test
#     -h | --help    show this help
#
# The reproduction scripts import the fractional_pareto package. Run this from
# the package (so its parent directory is the package root), or set the
# environment variable FRACTIONAL_PARETO_DIR to the package directory. The
# wrapper puts that directory on PYTHONPATH automatically and can be invoked
# from any working directory.
#
set -euo pipefail

usage() { sed -n '3,36p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'; }

# Resolve the directory of this script and the repository root (its parent).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FIGDIR="results/figures"
OUTDIR="results"
JOBS="${FPDE_REPRO_JOBS:-2}"
FAST=0
SMOKE=0
MODE="publication budget (5 seeds, DE 7x24, noise 0/1/5/10/20%)"

missing_value() {
  echo "error: $1 requires a value." >&2
  usage >&2
  exit 2
}

validate_jobs() {
  case "$JOBS" in
    ''|*[!0-9]*)
      echo "error: --jobs must be a positive integer; received '$JOBS'." >&2
      exit 2
      ;;
  esac
  if [ "$JOBS" -lt 1 ]; then
    echo "error: --jobs must be at least 1; received '$JOBS'." >&2
    exit 2
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --fast)
      FAST=1
      MODE="fast budget (reduced seeds/iterations)"
      shift
      ;;
    --smoke)
      FAST=1
      SMOKE=1
      MODE="smoke test (fast, Burgers only)"
      shift
      ;;
    -j|--jobs)
      [ $# -ge 2 ] || missing_value "$1"
      JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#*=}"
      shift
      ;;
    --figdir)
      [ $# -ge 2 ] || missing_value "$1"
      FIGDIR="$2"
      shift 2
      ;;
    --outdir)
      [ $# -ge 2 ] || missing_value "$1"
      OUTDIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
validate_jobs

# Package directory for imports: explicit override, else the repository root.
PKG_ROOT="${FRACTIONAL_PARETO_DIR:-$REPO_ROOT}"
export PYTHONPATH="${PKG_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Parallelise independent runs, not BLAS internals. This avoids severe CPU
# oversubscription on laptops when two or more discovery workers are active.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
cd "${REPO_ROOT}"

echo "=================================================================="
echo " fractional_pareto -- regenerate all tables and figures"
echo "   package root : ${PKG_ROOT}"
echo "   budget       : ${MODE}"
echo "   workers      : ${JOBS}"
echo "   tables  ->     ${OUTDIR}"
echo "   figures ->     ${FIGDIR}"
echo "=================================================================="

# --- dependency checks -------------------------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "error: python3 not found." >&2; exit 1; }
if ! python3 -c "import dataset_configs" >/dev/null 2>&1; then
  echo "error: cannot import the fractional_pareto package." >&2
  echo "       Run this from the package root, or set FRACTIONAL_PARETO_DIR" >&2
  echo "       to the directory that contains dataset_configs.py." >&2
  exit 1
fi
python3 -c "import matplotlib" >/dev/null 2>&1 || {
  echo "warning: matplotlib is not installed; the figure step will fail." >&2
  echo "         install it first, e.g.  pip install matplotlib" >&2
}

mkdir -p "${OUTDIR}" "${FIGDIR}"
export FPDE_REPRO_CACHE="${FPDE_REPRO_CACHE:-${OUTDIR}/.discovery_cache}"
export FPDE_REPRO_JOBS="${JOBS}"

if [ "${SMOKE}" -eq 1 ]; then
  echo
  echo ">>> [smoke 1/2] Running the test suite ..."
  python3 -u -m pytest -q
  echo
  echo ">>> [smoke 2/2] Running tiny Burgers weak/strong discovery ..."
  python3 -u "${SCRIPT_DIR}/smoke_burgers.py" --outdir "${OUTDIR}" --figdir "${FIGDIR}"
  echo
  echo ">>> Smoke mode: skipping publication-scale stages:"
  echo "    - shared campaign precomputation"
  echo "    - publication table and figure generation"
  echo "    - weak-row (K) sensitivity"
  echo "    - forward-model validation"
  echo "    - representative-equation generation"
  echo "    - component ablation"
  echo
  echo "Smoke workflow completed successfully (tests + tiny Burgers weak/strong discovery)."
  exit 0
fi

# Build each command from a nonempty array. This is portable to the older Bash
# versions whose `set -u` handling treats an empty `${array[@]}` expansion as an
# unbound variable (the cause of the former BUDGET_ARGS[@] failure).

# --- 0/7: shared run cache ---------------------------------------------------
echo
echo ">>> [0/7] Precomputing shared discovery runs (jobs=${JOBS}, resumable cache) ..."
SECONDS=0
cmd=(python3 -u "${SCRIPT_DIR}/precompute_campaign.py")
[ "${FAST}" -eq 0 ] || cmd+=(--fast)
cmd+=(--jobs "${JOBS}")
"${cmd[@]}"
echo ">>> shared precompute finished in ${SECONDS}s"

# --- 1/7: tables -------------------------------------------------------------
echo
echo ">>> [1/7] Generating tables from the shared cache ..."
SECONDS=0
cmd=(python3 -u "${SCRIPT_DIR}/make_tables.py")
[ "${FAST}" -eq 0 ] || cmd+=(--fast)
cmd+=(--outdir "${OUTDIR}")
"${cmd[@]}"
echo ">>> tables finished in ${SECONDS}s"

# --- 2/7: figures (Figures 1-3) ---------------------------------------------
echo
echo ">>> [2/7] Generating figures ..."
SECONDS=0
cmd=(python3 -u "${SCRIPT_DIR}/make_figures.py")
[ "${FAST}" -eq 0 ] || cmd+=(--fast)
cmd+=(--figdir "${FIGDIR}")
"${cmd[@]}"
echo ">>> figures finished in ${SECONDS}s"

# --- 3/7: weak-row (K) sensitivity + conditioning (Appendix F) --------------
echo
echo ">>> [3/7] Weak-row sensitivity (Appendix F, Table F2) ..."
SECONDS=0
cmd=(python3 -u "${SCRIPT_DIR}/make_ksens.py")
[ "${FAST}" -eq 0 ] || cmd+=(--fast)
cmd+=(--jobs "${JOBS}" --outdir "${OUTDIR}")
"${cmd[@]}"
echo ">>> K-sensitivity finished in ${SECONDS}s"

# --- 4/7: forward-model validation (Appendix G) -----------------------------
echo
echo ">>> [4/7] Forward-model validation (Appendix G, Table G3) ..."
SECONDS=0
cmd=(python3 -u "${SCRIPT_DIR}/make_forward.py")
[ "${FAST}" -eq 0 ] || cmd+=(--fast)
cmd+=(--outdir "${OUTDIR}")
"${cmd[@]}"
echo ">>> forward validation finished in ${SECONDS}s"

# --- 5/7: representative discovered equations (Appendix H) -----------------
echo
echo ">>> [5/7] Representative discovered equations (Appendix H, Table H4) ..."
SECONDS=0
cmd=(python3 -u "${SCRIPT_DIR}/make_equations.py")
[ "${FAST}" -eq 0 ] || cmd+=(--fast)
cmd+=(--outdir "${OUTDIR}")
"${cmd[@]}"
echo ">>> discovered equations finished in ${SECONDS}s"

# --- 6/7: component ablation on FADE ----------------------------------------
echo
echo ">>> [6/7] Component ablation on FADE (weak library / continuous orders / polishing) ..."
SECONDS=0
cmd=(python3 -u "${SCRIPT_DIR}/make_ablation.py")
[ "${FAST}" -eq 0 ] || cmd+=(--fast)
cmd+=(--outdir "${OUTDIR}")
"${cmd[@]}"
echo ">>> ablation finished in ${SECONDS}s"

# --- summary -----------------------------------------------------------------
echo
echo "=================================================================="
echo " All done."
echo " Tables (CSV data + .tex rows to paste into main.tex):"
ls -1 "${OUTDIR}"/table_*.csv "${OUTDIR}"/table_*.tex 2>/dev/null | sed 's/^/   /' || true
echo " Figures:"
ls -1 "${FIGDIR}"/*.png 2>/dev/null | sed 's/^/   /' || true
echo
echo " Next: paste each .tex file's rows into the matching tabular* in"
echo " main.tex, and (re)run with --figdir pointing at the manuscript's"
echo " figures/ directory to replace the placeholder figures in place."
echo "=================================================================="
