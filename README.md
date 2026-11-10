# Cluster-out cross-validation in crossed Gaussian models

This repository contains the code, data provenance and numerical results for local post-fit bounds on held-out squared loss. The examples compare predictions for a new group by withholding all observations belonging to that group.

| Experiment or check | How to reproduce it | Recorded results |
| --- | --- | --- |
| Training-only refits and timing | [Timing protocol](code/REPRODUCE_training_timing.txt) | [Timing plan, runs and summaries](results/training_timing/) |
| Lexical-decision post-fit intervals | [Post-fit validation](code/REPRODUCE_postfit.txt) | [Interval and fold outputs](results/postfit_validation/) |
| Independent random slopes, starting values and external timing | [Sensitivity study](code/REPRODUCE_revision.txt) | [Fit traces, comparisons and numerical checks](results/revision_study/) |
| Full-data-start benchmark | [Archived benchmark](code/REPRODUCE.txt) | [Precision, containment, model comparisons and timing](results/accuracy_v2/summary/) |
| OrchardSprays boundary case | [Data and commands](experiments/orchard/) | [Fold-level losses, first-test bounds and 60-digit checks](experiments/orchard/results/) |

The original benchmark and the training-only post-fit experiments use different starts. Their output directories and timing denominators are identified separately above. Dataset provenance is recorded with each dataset. The original solver license is in [code/LICENSE](code/LICENSE).
