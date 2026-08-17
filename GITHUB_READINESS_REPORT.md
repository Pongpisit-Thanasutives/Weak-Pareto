# GitHub readiness review - 10 August 2026

## Completed

- End-to-end tutorial script and executed notebook are included.
- Figure/table-to-script and public API documentation are included.
- Deterministic publication-asset renderers and regression tests are included.
- `.gitignore` and `CITATION.cff` use the planned repository URL.
- The Table-4 reproduction path now includes 0%, 2%, 5%, and 10% Riesz rows.
- The new ten-run clean Table-4 archive is included.
- Full test suite: 80/80 passed.
- Python syntax and all shell-script syntax checks passed.
- Runtime caches, Python bytecode, pytest caches, notebook checkpoints, and local
  temporary results are removed from release archives.

## Main entry points

- Tutorial script: `examples/tutorial_weak_pareto.py`
- Tutorial notebook: `notebooks/05_end_to_end_weak_pareto_tutorial.ipynb`
- Paper result index: `docs/PAPER_RESULTS_GUIDE.md`
- Public API guide: `docs/CODE_DOCUMENTATION.md`
- Full reproduction: `reproduce/README.md`
- Table 4 only: `reproduce/make_tables.py --only rd_noise`

## One author decision remains

No open-source license was selected automatically. Before making the repository
public, add the license you intend to grant and confirm that it is compatible
with every redistributed component and dataset.
