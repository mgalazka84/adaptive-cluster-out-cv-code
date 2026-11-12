# Training-only audits of cluster-out refits

These scripts reproduce the independent package benchmark, restart comparison and bound diagnostics. They use the data and solvers already included in this repository. Each training sample determines its covariance fits. A held-out response is used to score its prediction and to construct the loss interval at a fitted center.

The external benchmark used Python 3.12.14, statsmodels 0.14.6, NumPy 2.3.5, SciPy 1.17.0 and pandas 2.2.3. The cross-classified specification uses one top-level group and two independent variance components. The default benchmark calls `MixedLM.fit()` with no arguments, uses the returned fixed and random effects for prediction, and sends the reported variances directly to the local test. It keeps the optimizer convergence flag separate from warning text. The independent strict fit, deposited in `results/lexdec/statsmodels_audit.csv` at repository root, uses a stricter BFGS configuration in the same package version.

Run from this directory in that Python environment with Matplotlib installed.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python default_fits.py --mode accuracy
python default_fits.py --mode timing
python repair_timing.py
python restarts.py
python training_diagnostics.py
python boundary_geometry.py
python summarize_evidence.py
python figures.py
```

| Scientific question | Output in `results/` |
| --- | --- |
| Does a default package fit pass on its first check, and how many projected Newton updates are needed if it fails? | `default_folds.csv`, `default_summary.json`, `default_aggregate.csv` |
| What does the check cost relative to fitting and predicting? | `default_timing.csv`, `orchard_repair_timing.csv` |
| Do independent strict fits fall in the original, training-only lexical-decision intervals? | `strict_training_overlap.csv`, `evidence.json` |
| Do four starts identify the same local solution? Are the Latin-square component shapes independent after REML projection? | `restarts.csv`, `restarts_summary.json` |
| How do numerical half-widths compare with realized errors, and how do their sums compare with variation across folds? | `training_geometry.csv.gz`, `training_geometry.json`, `simulation_scales.csv`, `orchard_geometry.csv` |
| How are the two numerical figures made? | `Fig1.pdf`, `FigAuditGeometry.pdf` |

Timing uses the first, middle and last sorted group of each target and all three candidates (18 fits per dataset), one warm-up per dataset and three randomized repetitions. For each selected fit, take the median of its three package times and its three audit times, divide the latter by the former, and take the median of those paired ratios. The first-check timer includes construction of the compressed fold, Hessian and one local test. The separate OrchardSprays timing includes every repair and recheck through acceptance. Absolute times depend on the machine. Strict references and input slicing are outside the timers.

The loss containment join uses unique target/model/group keys, checks the original group labels and compares the raw interval endpoints without expanding them. Restart membership uses the free-coordinate uniqueness radius and equality of active sets. The geometry diagnostics discard loss errors below a fixed numerical resolution before forming the tightness ratio. The fold-variation proxy is descriptive because training folds overlap. Floating-point calculations use ordinary binary64 arithmetic.

## Additional diagnostics

Run second_round_checks.py to evaluate a simple projected-gradient and free-Hessian diagnostic at the 48 saved default Latin-square estimates. It uses the original variance coordinates, a gradient threshold of 1e-6, strictly positive returned components as free coordinates, and a nonpositive minimum Hessian eigenvalue as the curvature flag. It also summarizes the two existing timing trials and counts the separate budget-sensitivity design. Outputs are orchard_basic_diagnostics.csv, budget_design.csv and second_round_summary.json.

Run make_schematics.py to generate the three conceptual diagrams with consistent panel labels. These drawings illustrate the prediction targets and mathematical bounds. Their coordinates are illustrative.
