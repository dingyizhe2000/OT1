# Entropic OT 100-Rep Summary

Scope: entropic OT baseline outputs only from `simulation_results/entropic_ot`.

## Validation

- Total scenarios found: 48
- Expected scenarios: 48 (`d = 5, 10`; `n = 100, 300, 500, 1000`; `measure = normal, t`; `ot_map = CDF, piecewise_linear, quadratic`)
- Model `.pkl` files: 4800
- Metadata JSON files: 4800
- Scenarios with fewer than 100 reps: 0
  - None
- Missing scenarios/issues: 0
  - None
- PNW epsilon rule with `epsilon_c=1.0`, `alpha=1.0`: yes
- `sinkhorn_log` used in all scenarios: yes
- `out_of_sample_target_dual_log_scaling` used in all scenarios: yes

## Warning Summary

- Scenarios with non-convergence warnings: 31
  - d=5, n=100, measure=normal, ot_map=quadratic: 35 / 100
  - d=5, n=100, measure=t, ot_map=piecewise_linear: 50 / 100
  - d=5, n=100, measure=t, ot_map=quadratic: 45 / 100
  - d=5, n=300, measure=normal, ot_map=piecewise_linear: 3 / 100
  - d=5, n=300, measure=normal, ot_map=quadratic: 40 / 100
  - d=5, n=300, measure=t, ot_map=piecewise_linear: 46 / 100
  - d=5, n=300, measure=t, ot_map=quadratic: 40 / 100
  - d=5, n=500, measure=normal, ot_map=piecewise_linear: 2 / 100
  - d=5, n=500, measure=normal, ot_map=quadratic: 44 / 100
  - d=5, n=500, measure=t, ot_map=piecewise_linear: 40 / 100
  - d=5, n=500, measure=t, ot_map=quadratic: 43 / 100
  - d=5, n=1000, measure=normal, ot_map=piecewise_linear: 3 / 100
  - d=5, n=1000, measure=normal, ot_map=quadratic: 53 / 100
  - d=5, n=1000, measure=t, ot_map=piecewise_linear: 43 / 100
  - d=5, n=1000, measure=t, ot_map=quadratic: 46 / 100
  - d=10, n=100, measure=normal, ot_map=piecewise_linear: 4 / 100
  - d=10, n=100, measure=normal, ot_map=quadratic: 51 / 100
  - d=10, n=100, measure=t, ot_map=piecewise_linear: 44 / 100
  - d=10, n=100, measure=t, ot_map=quadratic: 46 / 100
  - d=10, n=300, measure=normal, ot_map=piecewise_linear: 2 / 100
  - d=10, n=300, measure=normal, ot_map=quadratic: 39 / 100
  - d=10, n=300, measure=t, ot_map=piecewise_linear: 45 / 100
  - d=10, n=300, measure=t, ot_map=quadratic: 48 / 100
  - d=10, n=500, measure=normal, ot_map=piecewise_linear: 4 / 100
  - d=10, n=500, measure=normal, ot_map=quadratic: 45 / 100
  - d=10, n=500, measure=t, ot_map=piecewise_linear: 41 / 100
  - d=10, n=500, measure=t, ot_map=quadratic: 45 / 100
  - d=10, n=1000, measure=normal, ot_map=piecewise_linear: 4 / 100
  - d=10, n=1000, measure=normal, ot_map=quadratic: 52 / 100
  - d=10, n=1000, measure=t, ot_map=piecewise_linear: 45 / 100
  - d=10, n=1000, measure=t, ot_map=quadratic: 48 / 100
- Scenarios with overflow warnings: 22
  - d=5, n=100, measure=t, ot_map=piecewise_linear: 1 / 100
  - d=5, n=100, measure=t, ot_map=quadratic: 40 / 100
  - d=5, n=300, measure=t, ot_map=piecewise_linear: 2 / 100
  - d=5, n=300, measure=t, ot_map=quadratic: 40 / 100
  - d=5, n=500, measure=normal, ot_map=quadratic: 2 / 100
  - d=5, n=500, measure=t, ot_map=CDF: 1 / 100
  - d=5, n=500, measure=t, ot_map=piecewise_linear: 2 / 100
  - d=5, n=500, measure=t, ot_map=quadratic: 43 / 100
  - d=5, n=1000, measure=normal, ot_map=quadratic: 6 / 100
  - d=5, n=1000, measure=t, ot_map=CDF: 3 / 100
  - d=5, n=1000, measure=t, ot_map=piecewise_linear: 4 / 100
  - d=5, n=1000, measure=t, ot_map=quadratic: 46 / 100
  - d=10, n=100, measure=t, ot_map=piecewise_linear: 1 / 100
  - d=10, n=100, measure=t, ot_map=quadratic: 44 / 100
  - d=10, n=300, measure=t, ot_map=piecewise_linear: 1 / 100
  - d=10, n=300, measure=t, ot_map=quadratic: 47 / 100
  - d=10, n=500, measure=normal, ot_map=quadratic: 1 / 100
  - d=10, n=500, measure=t, ot_map=piecewise_linear: 3 / 100
  - d=10, n=500, measure=t, ot_map=quadratic: 45 / 100
  - d=10, n=1000, measure=normal, ot_map=quadratic: 1 / 100
  - d=10, n=1000, measure=t, ot_map=piecewise_linear: 1 / 100
  - d=10, n=1000, measure=t, ot_map=quadratic: 48 / 100
- Scenarios with other warnings: 0

## Compact L2 Table

| ot_map | measure | d | n | reps | mean | median | sd | p25 | p75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CDF | normal | 5 | 100 | 100 | 0.123576 | 0.122743 | 0.005905 | 0.120321 | 0.127461 |
| piecewise_linear | normal | 5 | 100 | 100 | 0.453867 | 0.448059 | 0.027321 | 0.433603 | 0.468770 |
| quadratic | normal | 5 | 100 | 100 | 1.092272 | 1.080993 | 0.084201 | 1.037177 | 1.136299 |
| CDF | t | 5 | 100 | 100 | 0.121036 | 0.121438 | 0.006561 | 0.116840 | 0.124433 |
| piecewise_linear | t | 5 | 100 | 100 | 0.872504 | 0.822260 | 0.146060 | 0.786119 | 0.926532 |
| quadratic | t | 5 | 100 | 100 | 3.158439 | 2.826923 | 1.037128 | 2.596240 | 3.175830 |
| CDF | normal | 5 | 300 | 100 | 0.099354 | 0.099394 | 0.003361 | 0.097077 | 0.101433 |
| piecewise_linear | normal | 5 | 300 | 100 | 0.341902 | 0.340275 | 0.015043 | 0.332572 | 0.350443 |
| quadratic | normal | 5 | 300 | 100 | 0.846484 | 0.840707 | 0.050630 | 0.809100 | 0.876582 |
| CDF | t | 5 | 300 | 100 | 0.095336 | 0.095127 | 0.003373 | 0.093170 | 0.097342 |
| piecewise_linear | t | 5 | 300 | 100 | 0.657486 | 0.641814 | 0.061821 | 0.612337 | 0.683638 |
| quadratic | t | 5 | 300 | 100 | 2.496291 | 2.332761 | 0.596227 | 2.191979 | 2.605501 |
| CDF | normal | 5 | 500 | 100 | 0.091630 | 0.091564 | 0.002605 | 0.090298 | 0.092972 |
| piecewise_linear | normal | 5 | 500 | 100 | 0.299382 | 0.298147 | 0.011723 | 0.290748 | 0.306433 |
| quadratic | normal | 5 | 500 | 100 | 0.750648 | 0.742459 | 0.041987 | 0.722986 | 0.771958 |
| CDF | t | 5 | 500 | 100 | 0.086921 | 0.087184 | 0.002991 | 0.085048 | 0.088761 |
| piecewise_linear | t | 5 | 500 | 100 | 0.576648 | 0.562861 | 0.045926 | 0.543408 | 0.595875 |
| quadratic | t | 5 | 500 | 100 | 2.314593 | 2.135283 | 0.609387 | 2.003945 | 2.462068 |
| CDF | normal | 5 | 1000 | 100 | 0.082223 | 0.082108 | 0.002083 | 0.080655 | 0.083893 |
| piecewise_linear | normal | 5 | 1000 | 100 | 0.250474 | 0.249398 | 0.007947 | 0.244752 | 0.254776 |
| quadratic | normal | 5 | 1000 | 100 | 0.631577 | 0.626940 | 0.033766 | 0.610916 | 0.642196 |
| CDF | t | 5 | 1000 | 100 | 0.077698 | 0.077851 | 0.002009 | 0.076209 | 0.079265 |
| piecewise_linear | t | 5 | 1000 | 100 | 0.494187 | 0.488846 | 0.039338 | 0.469221 | 0.508720 |
| quadratic | t | 5 | 1000 | 100 | 2.027274 | 1.902952 | 0.429939 | 1.781506 | 2.103090 |
| CDF | normal | 10 | 100 | 100 | 0.162123 | 0.161645 | 0.003032 | 0.160473 | 0.163813 |
| piecewise_linear | normal | 10 | 100 | 100 | 0.643403 | 0.640404 | 0.017777 | 0.631377 | 0.651503 |
| quadratic | normal | 10 | 100 | 100 | 1.396994 | 1.390364 | 0.083483 | 1.343945 | 1.434052 |
| CDF | t | 10 | 100 | 100 | 0.163458 | 0.163469 | 0.002636 | 0.161592 | 0.165245 |
| piecewise_linear | t | 10 | 100 | 100 | 1.060440 | 1.040916 | 0.080669 | 1.014286 | 1.093116 |
| quadratic | t | 10 | 100 | 100 | 3.327911 | 3.117372 | 1.012078 | 2.965399 | 3.320189 |
| CDF | normal | 10 | 300 | 100 | 0.136478 | 0.136312 | 0.002041 | 0.135355 | 0.137815 |
| piecewise_linear | normal | 10 | 300 | 100 | 0.539075 | 0.538099 | 0.008389 | 0.533739 | 0.544630 |
| quadratic | normal | 10 | 300 | 100 | 1.193551 | 1.187766 | 0.036827 | 1.170029 | 1.213612 |
| CDF | t | 10 | 300 | 100 | 0.135352 | 0.135385 | 0.001989 | 0.134194 | 0.136635 |
| piecewise_linear | t | 10 | 300 | 100 | 0.890935 | 0.882897 | 0.043043 | 0.861155 | 0.906292 |
| quadratic | t | 10 | 300 | 100 | 2.934877 | 2.761721 | 0.549247 | 2.611687 | 3.060043 |
| CDF | normal | 10 | 500 | 100 | 0.127330 | 0.127441 | 0.001413 | 0.126330 | 0.128430 |
| piecewise_linear | normal | 10 | 500 | 100 | 0.496225 | 0.496729 | 0.007088 | 0.492458 | 0.500256 |
| quadratic | normal | 10 | 500 | 100 | 1.100297 | 1.099315 | 0.025569 | 1.077400 | 1.117034 |
| CDF | t | 10 | 500 | 100 | 0.125585 | 0.125392 | 0.001434 | 0.124433 | 0.126607 |
| piecewise_linear | t | 10 | 500 | 100 | 0.821786 | 0.815492 | 0.032602 | 0.798223 | 0.839699 |
| quadratic | t | 10 | 500 | 100 | 2.734684 | 2.592889 | 0.592786 | 2.413160 | 2.847011 |
| CDF | normal | 10 | 1000 | 100 | 0.117578 | 0.117465 | 0.001239 | 0.116773 | 0.118236 |
| piecewise_linear | normal | 10 | 1000 | 100 | 0.441387 | 0.440870 | 0.004787 | 0.438305 | 0.443489 |
| quadratic | normal | 10 | 1000 | 100 | 0.993376 | 0.990061 | 0.020427 | 0.983418 | 0.999824 |
| CDF | t | 10 | 1000 | 100 | 0.114151 | 0.114067 | 0.001157 | 0.113470 | 0.114773 |
| piecewise_linear | t | 10 | 1000 | 100 | 0.739266 | 0.735614 | 0.025939 | 0.719912 | 0.750820 |
| quadratic | t | 10 | 1000 | 100 | 2.399176 | 2.301279 | 0.425222 | 2.149817 | 2.487425 |

Generated with bootstrap seed `20260527` and 2000 bootstrap resamples for `bootstrap_sd_mean` and `bootstrap_sd_median`.
