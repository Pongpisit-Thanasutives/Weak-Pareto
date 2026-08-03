# Frozen-soil creep data (external)

The real-data example in `real_data/frozen_soil_creep_weak.py` expects two
MATLAB files:

- `creep.mat` (clay; variables `t` and `ex`)
- `silt.mat` (silt; variables `t` and `ex`)

These irregular, naturally noisy strain histories are the frozen-soil creep
data used by Yu et al., *A data-driven framework for discovering fractional
differential equations in complex systems*, **Nonlinear Dynamics** 113 (2025),
24557-24577, DOI 10.1007/s11071-025-11373-z.

The files are not redistributed in this package because the upstream source
does not provide a clear data-redistribution licence. Download the files from the authors' public repository:

```text
https://github.com/yxn1019/FDE_discovery/tree/main/dataset
```

Copy the two files into this directory and run:

```bash
PYTHONPATH=. python real_data/frozen_soil_creep_weak.py \
  --data-dir external_data/frozen_soil \
  --outdir results/frozen_soil_creep
```

For provenance checks, the files used for the manuscript had SHA-256 hashes:

```text
creep.mat  15f8d1a8b8f4eeb1736b2f7ff5976b93bed022971183b5f9de992bcb31033fff
silt.mat   63d556ec162231c3e4a57552e57ff300843a3d490f2e8561da2744152596f01d
```

The example identifies the fractional order and Kelvin parameters within the
physically motivated support `{1, epsilon}`. It is not presented as an
unrestricted structure-discovery result.
