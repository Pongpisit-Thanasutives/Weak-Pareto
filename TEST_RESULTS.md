# Final verification results

The exact release tree was checked in the final presubmission audit with:

```text
complete pytest suite                                  80/80 passed
documentation audit                                    passed (53 non-test modules)
Python syntax/bytecode compilation                     passed
bash -n on every shell script                          passed
Burgers weak/strong smoke workflow                     passed
superunit clean/noisy smoke workflow                   passed
complete reproduce/run_all.sh --smoke workflow        passed
concise tutorial script                                passed
matching tutorial notebook                             executed successfully
```

The tutorial selected the intended two-term FADE support and returned

```text
D_t^0.79950 u = (-1.0068)*D_x^1.00000 u + (0.49614)*D_x^1.70000 u
```

## Figure regression checks

```text
Figure 1 values are loaded from archived publication outputs     passed
Figure 1 Gaussian field uses the published noise definition      passed
Figure 2 curves match table_robustness.csv                        passed
Figure 3 curve matches table_progress.csv                         passed
Figure 3 uses the intended linear y-axis                          passed
Figure 4 contains all six tested noise levels                     passed
Figure 4 full curve recomputes from the discovery code            passed
Figure 4 agrees with Table 8 at 0%, 10%, and 25%                  passed
Figure 4 margins remain >1 and coefficient errors remain <0.02    passed
```

The six-point Burgers curve was independently recomputed from the released
weak-library and fixed-structure fitting code.  Its archived residuals agree to
floating-point precision.  The reference-output archive otherwise remains
byte-for-byte unchanged from the stable pre-conversion package.

## Release-integrity checks

- Core discovery modules, dataset definitions, and pre-existing archived results
  were compared by SHA-256 with the stable pre-conversion package.
- The manuscript was compiled twice from a clean directory using the included
  `main.bbl`.
- The final PDF contains 40 pages, embedded fonts, no undefined citations or
  references, no duplicate labels, no missing graphics, and no overfull boxes.
- Figure PDFs are vector artwork except for the explicitly rasterised solution
  fields, which are embedded at 600 dpi.
