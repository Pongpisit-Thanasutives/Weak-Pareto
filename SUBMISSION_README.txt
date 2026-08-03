ONLINE RESOURCE 1

Article title: Robust data-driven discovery of fractional differential equations via weak formulations and Pareto-based subset selection
Journal: Nonlinear Dynamics
Authors: Pongpisit Thanasutives and Yoshinobu Kawahara
Affiliations:
  1. Center for Advanced Intelligence Project (AIP), RIKEN, Tokyo, Japan
  2. Graduate School of Information Science and Technology, The University of Osaka, Suita, Japan
Corresponding author: Pongpisit Thanasutives <pongpisit.thanasutives@riken.jp>
Repository: https://github.com/Pongpisit-Thanasutives/Weak-Pareto

This archive contains the Weak-Pareto implementation, benchmark-data utilities, reproduction workflows, tests, documentation, notebooks, and the aggregate reference outputs used in the final manuscript. Pareto selection uses variance-normalized validation error (val_rel_mse) throughout. The exact-order/full-data residual is stored separately as full_data_rel_l2.

Final reference outputs:
  reproduce/reference_results/branch_aware_campaign/
  reproduce/reference_results/yu_framework_comparison/
  reproduce/reference_results/superunit_final/
  two_dimensional/reference_results/
  two_dimensional/resolution_112/results/

First-use tutorial:
  python examples/tutorial_weak_pareto.py
  jupyter lab notebooks/05_end_to_end_weak_pareto_tutorial.ipynb

Paper object-to-script index:
  docs/PAPER_RESULTS_GUIDE.md

Verification commands:
  python scripts/check_documentation.py
  python -m compileall -q .
  pytest -q
  bash reproduce/run_all.sh --smoke --jobs 2
  PYTHONPATH=. python reproduce/smoke_superunit.py --outdir results/superunit_smoke

Full campaign reproduction:
  bash run_branch_aware_campaign.sh results/normalized_metric_final 5

For the internal campaign alone, the worker count can be set with --jobs N, --jobs=N, -j N, or FPDE_REPRO_JOBS=N. The upstream source for the Yu et al. neural fractional-discovery framework and frozen-soil raw files are not redistributed; see the corresponding documentation.

Superunit fixed-support branch/order experiment (reported in the manuscript):
  PYTHONPATH=. python reproduce/run_superunit_experiment.py --outdir results/superunit_final --jobs 5

This diagnostic fixes the one-term linear support and evaluates branch/order
recovery from semi-analytic data at 0%, 0.5%, and 1% noise.

Two-dimensional example:
  cd two_dimensional
  ./run_all.sh publication

This workflow reproduces two dense periodic directional benchmarks on a
90 x 80 x 80 grid, with five seeds and noise through 20%. The 48,000 weak
rows are overlapping tensor-product integral equations, not independent data.
Sparse and nonlinear 2D discovery are outside the present claim.
