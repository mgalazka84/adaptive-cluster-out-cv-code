"""Real-data computational case study with two distinct prediction targets.

All 1,659 supplied observations are retained. RT is already logarithmic.
The candidate designs use only covariates known before the current response.
Trial/100 is a fixed change of units, not a learned transformation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from acv import CompressedFold, make_fold
from boundary_acv import fit_box
from run_boundary_study import benchmark

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]
MODEL_NAMES = ['Trial + Frequency', '+ Length + NativeLanguage', '+ Length:NativeLanguage']


def load_data():
    raw = ROOT/'data'/'lexdec.csv'
    provenance = json.loads((ROOT/'data'/'lexdec_provenance.json').read_text())
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == provenance['csv_sha256']
    df = pd.read_csv(raw, float_precision='round_trip')
    assert len(df) == 1659 and df.Subject.nunique() == 21 and df.Word.nunique() == 79
    assert not df[['RT', 'Trial', 'Frequency', 'Length', 'NativeLanguage', 'Subject', 'Word']].isna().any().any()
    assert df.groupby(['Subject', 'Word']).size().eq(1).all()
    other = (df.NativeLanguage == 'Other').astype(float).to_numpy()
    X = np.column_stack([np.ones(len(df)), df.Trial/100, df.Frequency,
                         df.Length, other, df.Length*other])
    return df, X, df.RT.to_numpy()


def make_problem(target):
    df, X, y = load_data()
    first, second = ('Subject', 'Word') if target == 'new_subject' else ('Word', 'Subject')
    a = df[first].to_numpy(); b = df[second].to_numpy()
    t0 = perf_counter()
    scale = float(np.var(y))
    starts = [fit_box(CompressedFold(X[:, cols], y, a, b), scale*np.array([.5, .25, .25]))[0]
              for cols in COLUMNS]
    groups = np.unique(a)
    folds = [[make_fold(X, y, a, b, cols, group, 'compressed') for group in groups] for cols in COLUMNS]
    return folds, starts, perf_counter()-t0, groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timing-reps', type=int, default=7)
    args = p.parse_args()
    out = ROOT/'results'/'lexdec'; out.mkdir(parents=True, exist_ok=True)
    rows, audits, timings = [], [], []
    for i, target in enumerate(['new_subject', 'new_word']):
        folds, starts, setup, groups = make_problem(target)
        row, ar, tr = benchmark(folds, starts, setup, 88001+i, args.timing_reps)
        row.update(dataset='lexdec', target=target, n=1659, groups=len(groups))
        rows.append(row)
        audits.extend(dict(target=target, group=str(groups[x['group_index']]), **x) for x in ar)
        timings.extend(dict(target=target, **x) for x in tr)
        pd.DataFrame(rows).to_csv(out/'runs.csv', index=False)
        pd.DataFrame(audits).to_csv(out/'fold_audit.csv', index=False)
        pd.DataFrame(timings).to_csv(out/'timings.csv', index=False)
        print(json.dumps(row, indent=2, default=float), flush=True)
    design = {'dataset': 'languageR::lexdec', 'observations': 1659,
              'response': 'Supplied RT = logarithmically transformed reaction time; no further response transform.',
              'preprocessing': 'All supplied rows; no outcome trimming, imputation, or learned normalization.',
              'columns': ['Intercept', 'Trial / 100', 'Frequency', 'Length', 'NativeLanguageOther', 'Length * NativeLanguageOther'],
              'candidate_column_indices': COLUMNS, 'model_names': MODEL_NAMES,
              'weights': 'Equal held-out-group weights; complete 21 by 79 crossed layout.',
              'random_effects': 'Independent subject and word random intercepts in all candidates.',
              'timing_repeats': args.timing_reps,
              'scope': 'One computational case study with two prediction tasks, not two independent datasets; no causal or population-best-model claim.',
              'prospective_design': 'Candidate formulas, all-row inclusion, two targets, and timing repetitions fixed before viewing CV results in this development stage; no external preregistration.'}
    (out/'design.json').write_text(json.dumps(design, indent=2))


if __name__ == '__main__':
    main()
