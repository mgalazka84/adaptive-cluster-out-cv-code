# OrchardSprays boundary experiment

The 64 observed responses form an 8-by-8 Latin square. On the log-volume scale the three fixed-effect designs use one common mean, a linear dilution trend with a separate control mean, or eight treatment means. All fits allow random row and column intercepts. Leaving out each of eight rows and then each of eight columns gives 48 model-by-fold fits.

## Reproduce the analysis

From this directory, install the pinned dependencies in a Python 3.12 environment and run the three commands below. The recorded environment used Python 3.12.14 and one BLAS thread. On a POSIX shell run `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` first. In PowerShell set each corresponding environment variable to `"1"`.

```sh
python -m pip install -r requirements.txt
python run_orchard.py --output replay_original
python check_local_bounds.py --output replay_dense
python verify_orchard.py --recorded replay_original/orchard_folds.csv --output replay_60
```

The first command after installation runs the post-fit algorithm from [the repository's solver directory](../../code/). The second repeats the practical fits and first local tests using full error contrasts. The last computes a 60-digit reference on the proposed zero-column-variance face, checks the one-sided full-model score conditions and compares each loss with the original raw endpoints. Without `--recorded`, it compares with the committed fold outputs. It uses the recorded 64 binary64 log responses so that comparisons at the last bit refer to the same inputs.

## Results

- [Mean scores and endpoints](results/orchard_scores.csv) and [all 48 post-fit records](results/orchard_folds.csv).
- [Results from the independent dense implementation](results/dense_replay.json) and [its fold records](results/dense_replay_folds.csv).
- [60-digit profile summary](results/orchard_profile60.json) and [the reference loss and endpoint comparison for each fold](results/orchard_profile60.csv).
- [Data source and checksums](SOURCE.txt). The data are distributed with R as `datasets::OrchardSprays`.

The fitted column variance was zero in all 48 fits. Seventeen fits also had zero row variance. Both target comparisons selected the treatment-means design after one local test per fit. All 16 intervals of positive width contained the independent 60-digit loss. In the other 32 fits the endpoints coincide in binary64 and differ from the 60-digit result by at most 9.06e-16. These raw endpoints have no allowance for floating-point rounding.
