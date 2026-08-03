# Paper result-to-script guide

This guide maps each numerical object in the manuscript to the command that
reproduces it and to the archived reference output shipped with the repository.
Run commands from the `Supplementary_Code/` directory unless stated otherwise.
The manuscript labels are more stable than table numbers, so both are shown.

## Main-text figures

| Manuscript object | LaTeX label | Publication-asset renderer | Full-result reproduction | Output |
|---|---|---|---|---|
| Figure 1: graphical overview | `fig:overview` | `python ../Manuscript/make_graphical_overview.py` | Uses the bundled FADE data and archived manuscript values | `../Manuscript/figures/Fig1.{pdf,png}` and `../Manuscript/GraphicalAbstract.{pdf,png}` |
| Figure 2: weak versus strong robustness | `fig:robustness` | `PYTHONPATH=. python reproduce/render_archived_paper_figures.py --only robustness --figdir ../Manuscript/figures` | `PYTHONPATH=. python reproduce/make_figures.py --only robustness --figdir ../Manuscript/figures` | `../Manuscript/figures/Fig2.{pdf,png}` |
| Figure 3: Pareto elbow | `fig:pareto` | `PYTHONPATH=. python reproduce/render_archived_paper_figures.py --only pareto --figdir ../Manuscript/figures` | `PYTHONPATH=. python reproduce/make_figures.py --only pareto --figdir ../Manuscript/figures` | `../Manuscript/figures/Fig3.{pdf,png}` |
| Figure 4: fractional Burgers field and residual separation | `fig:burgers` | `PYTHONPATH=. python reproduce/render_archived_paper_figures.py --only burgers --figdir ../Manuscript/figures` | `PYTHONPATH=. python reproduce/make_figures.py --only burgers --figdir ../Manuscript/figures` | `../Manuscript/figures/Fig4.{pdf,png}` |

The publication renderer is the fast deterministic route from the archived CSV
summaries. Figure 4 uses `figure_burgers_curve.csv`, which stores all six plotted
noise levels; `table_burgers.csv` remains the three-row summary used by Table 8.
The full-result script reruns the discovery or fixed-structure calculations before
writing the same manuscript-numbered vector PDF and 600-dpi PNG files. Figure 1 is generated directly from the supplied FADE data and validated archived outputs through `reproduce/graphical_overview_inputs.py`; it contains no generative imagery.

## Main-text and appendix tables

| Current table | LaTeX label | What it reports | Reproduction entry point |
|---|---|---|---|
| Table 1 | `tab:method-positioning` | Method-positioning summary | Manuscript-only literature synthesis; no numerical script |
| Table 2 | `tab:benchmarks` | Benchmark equations and grids | Definitions in `dataset_configs.py` and `fpde_datasets.py` |
| Table 3 | `tab:main` | Main 10% benchmark results | `PYTHONPATH=. python reproduce/make_tables.py --only main` |
| Table 4 | `tab:rdnoise` | Riesz reaction-diffusion versus noise | `PYTHONPATH=. python reproduce/make_tables.py --only rd_noise` |
| Table 5 | `tab:superunit` | Superunit Caputo diagnostic | `PYTHONPATH=. python reproduce/run_superunit_experiment.py --outdir results/superunit_final --jobs 5` |
| Table 6 | `tab:robustness` | Weak versus strong-form FADE comparison | `PYTHONPATH=. python reproduce/make_tables.py --only robustness` |
| Table 7 | `tab:progress` | Best equation at each support size | `PYTHONPATH=. python reproduce/make_tables.py --only progress` |
| Table 8 | `tab:burgers` | Nonlinear Burgers recovery margin | `PYTHONPATH=. python reproduce/make_tables.py --only burgers` |
| Table 9 | `tab:frozen-soil` | Frozen-soil Kelvin fits | `PYTHONPATH=. python real_data/frozen_soil_creep_weak.py --data-dir external_data/frozen_soil --outdir results/frozen_soil_creep` |
| Table 10 | `tab:twod` | Two-dimensional directional recovery | `cd two_dimensional && ./run_all.sh publication` |
| Table 11 | `tab:runtime` | Runtime and search budget | `PYTHONPATH=. python reproduce/make_tables.py --only runtime` |
| Table 12 | `tab:ablation` | Weak-library, continuous-order, and polishing ablation | `PYTHONPATH=. python reproduce/make_ablation.py` |
| Table 13 | `tab:conditioning` | Fixed-dictionary conditioning | `PYTHONPATH=. python reproduce/make_tables.py --only conditioning` |
| Table 14 | `tab:searchranges` | Declared search domains | `dataset_configs.py`; export with `python scripts/export_canonical_configs.py` |
| Table 15 | `tab:appendix` | Challenging ADE and two-term Riesz cases | `PYTHONPATH=. python reproduce/make_tables.py --only appendix` |
| Table 16 | `tab:ksens` | Sensitivity to weak-row count | `PYTHONPATH=. python reproduce/make_ksens.py` |
| Table 17 | `tab:gaussian-noise` | Additive-Gaussian robustness | `PYTHONPATH=. python scripts/run_alternative_robustness.py --conditions additive_gaussian --noise 10 --seeds 0 1 2 3 4 --restart --outdir results/alternative_robustness` |
| Table 18 | `tab:forward` | Forward-model validation | `PYTHONPATH=. python reproduce/make_forward.py` |
| Table 19 | `tab:equations` | Representative discovered equations | `PYTHONPATH=. python reproduce/make_equations.py` |
| Table 20 | `tab:twod-width` | Two-dimensional window-width sensitivity | `cd two_dimensional && ./run_all.sh publication` |
| Table 21 | `tab:twod-resolution` | Two-dimensional resolution check | See `two_dimensional/resolution_112/README.md` |
| Table 22 | `tab:yu` | Adapted Yu-framework comparison | `FPDE_DEVICE=cpu bash scripts/run_yu2025_comparison.sh` |

## Mathematical and diagnostic appendices

| Claim or diagnostic | Script |
|---|---|
| Proposition 1 variance scaling | `python reproduce/check_variance_scaling.py` |
| Riesz and Caputo operator convergence | `PYTHONPATH=. python reproduce/check_operator_convergence.py` |
| Nonlinear weak-feature bias | `PYTHONPATH=. python reproduce/check_nonlinear_bias.py` |
| Oracle temporal-order profiles | `PYTHONPATH=. python reproduce/profile_temporal_order.py` |

## Full campaigns

- Complete internal manuscript campaign: `reproduce/run_all.sh --jobs 2`
- Final branch-aware campaign, including additive Gaussian and Weak-Pareto rows
  for the neural comparison: `bash run_branch_aware_campaign.sh results/normalized_metric_final 5`
- Fast software check only: `reproduce/run_all.sh --smoke`

Archived publication outputs are under `reproduce/reference_results/`. Fast or
smoke profiles verify the software path but must not be used to replace the
reported publication numbers.
