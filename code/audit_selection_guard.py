"""Check archived final intervals against the numerical tie guard.

This reads prior results; it does not rerun optimization or benchmark time.
"""
import json
from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    records = []
    for study, keys in [('boundary_v2', ['regime', 'seed']),
                        ('boundary_scale_960', ['regime', 'seed']),
                        ('lexdec', ['target'])]:
        frame = pd.read_csv(root/'results'/study/'fold_audit.csv')
        for key, subset in frame.groupby(keys):
            scores = subset.groupby('model')[['lower', 'upper', 'reference_loss']].mean()
            best = int(scores['upper'].idxmin())
            other = float(scores.loc[scores.index != best, 'lower'].min())
            hi = float(scores.loc[best, 'upper'])
            tolerance = 1e-10+1e-10*max(abs(hi), abs(other))
            gap = other-hi
            reference = int(scores['reference_loss'].idxmin())
            assert gap > tolerance and best == reference, (study, key, gap, tolerance)
            records.append({'study': study, 'case': str(key), 'winner': best,
                            'gap': gap, 'tolerance': tolerance,
                            'gap_to_tolerance_ratio': gap/tolerance})
    summary = {
        'checked_archived_decisions': len(records), 'all_winners_retained': True,
        'minimum_gap': min(r['gap'] for r in records),
        'minimum_gap_to_tolerance_ratio': min(r['gap_to_tolerance_ratio'] for r in records),
        'scope': 'Audit of archived final intervals only; no new optimization or timing benchmark.',
        'records': records,
    }
    (root/'results'/'selection_guard_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
