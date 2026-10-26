"""Certified-output comparisons and timing resolution from retained runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from accuracy_io import read_csv

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'accuracy_v2'
OUT = BASE / 'summary'
SUITES = ['original', 'scale', 'fresh', 'lexdec']
R = pd.concat([read_csv(BASE / s / 'runs.csv') for s in SUITES], ignore_index=True)
groups = [('Development', R[R.suite == 'original']),
          ('Scale', R[R.suite == 'scale']),
          ('Fresh all', R[R.suite == 'fresh']),
          ('Fresh large', R[(R.suite == 'fresh') & (R.regime == 'many_levels')]),
          ('Lexdec participant', R[(R.suite == 'lexdec') & (R.regime == 'new_subject')]),
          ('Lexdec word', R[(R.suite == 'lexdec') & (R.regime == 'new_word')])]
rows = []
for label, d in groups:
    rows.append(dict(study=label, comparisons=len(d),
                     adaptive_kernel_ms=1000*d.adaptive_seconds.median(),
                     once_kernel_ms=1000*d.certify_once_seconds.median(),
                     kernel_paired_ratio=(d.certify_once_seconds/d.adaptive_seconds).median(),
                     total_paired_ratio=((d.common_seconds+d.certify_once_seconds)/
                                         (d.common_seconds+d.adaptive_seconds)).median(),
                     once_resolved=int(d.cert_once_resolves.sum())))
pd.DataFrame(rows).to_csv(OUT / 'certified_comparison.csv', index=False)
timing = pd.concat([read_csv(BASE / s / 'timings.csv') for s in SUITES], ignore_index=True)
spread = timing.groupby(['suite', 'regime', 'seed', 'method']).seconds.agg(['min', 'max', 'median'])
spread['relative_range'] = (spread['max']-spread['min'])/spread['median']
spread.reset_index().to_csv(OUT / 'timing_repetition_spread.csv', index=False)
stats = dict(comparisons=len(R),
             practical_faster=int((R.practical_speedup_total < 1).sum()),
             default_faster=int((R.default_speedup_total < 1).sum()),
             different_faster_indicators=int(((R.practical_speedup_total < 1) !=
                                             (R.default_speedup_total < 1)).sum()))
for name, col in [('practical', 'practical_speedup_total'), ('default', 'default_speedup_total')]:
    stats[name+'_total_ratio_within_five_percent'] = int((abs(R[col]-1) < .05).sum())
for name in ['adaptive', 'practical', 'default_lbfgs', 'certify_once']:
    d = spread.xs(name, level='method').relative_range
    stats[name+'_repetition_relative_range_median'] = float(d.median())
    stats[name+'_repetition_relative_range_p95'] = float(d.quantile(.95))
(OUT / 'timing_resolution.json').write_text(json.dumps(stats, indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(stats, indent=2))
