# Operator-structure tolerance sensitivity

This directory records a post hoc sensitivity analysis of the binary operator-structure recovery count. It does not rerun discovery or change any selected model.

Each row in `per_run.csv` is one noisy run from the matched Weak-Pareto and Strong Pareto comparisons: multiplicative-uniform FADE and Burgers experiments at 1%, 5%, 10%, and 20% noise; the 10% additive-Gaussian FADE and Burgers experiments; and the 0.5% and 1% superunit diagnostic. There are 60 runs per method.

Recovery always requires the correct support and powers, the correct temporal mode, exact recovery of identity terms, and a maximum absolute error below the stated tolerance for every positive temporal or spatial derivative order. `summary.csv` reports the resulting totals.

The manuscript uses 0.15 as a predefined operational tolerance, not as a statistically optimal cutoff. The sensitivity totals show that the main weak-versus-strong conclusion is stable around this value.
