"""Reproduce the post-fit lexdec intervals for the revised application table.

Same data, designs and starts as the archived training-only experiment.
This is an untimed deterministic replay, not a new timing benchmark.
Original timing outputs and plans are not modified.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_training_timing import prepare, run

ROOT = Path(__file__).resolve().parents[1]


def main():
    archived = pd.read_csv(ROOT / 'results/training_timing/folds.csv.gz')
    rows, summaries = [], []
    max_loss_difference = 0.
    for kind, seed in [('new_subject', 88001), ('new_word', 88002)]:
        folds, _ = prepare(kind, seed)
        result, strict = run('bound_once', folds), run('newton', folds)
        G = len(folds[0])
        previous = archived[archived.kind == kind].sort_values(['model', 'group'])
        for method, scores in [('bound_once', result['scores']), ('newton', strict['scores'])]:
            error = float(np.max(abs(scores.ravel() - previous[method + '_loss'].to_numpy())))
            assert error < 5e-12, (kind, method, error)
            max_loss_difference = max(max_loss_difference, error)
        assert result['bound_winner'] == strict['winner'] == 2
        for i, cert in enumerate(result['certs']):
            m, k = divmod(i, G)
            assert cert['valid'] and cert['reason'] == 'interior'
            free, active = cert['free'], cert['active']
            ref = strict['theta'][i]
            distance = np.linalg.norm(np.linalg.solve(cert['S'][free], (ref - cert['theta'])[free]))
            assert np.array_equal(np.flatnonzero(ref == 0), active)
            assert distance <= cert['outer_radius']
            loss = strict['scores'][m, k]
            assert cert['lo'] <= loss <= cert['hi']
            rows.append(dict(kind=kind, model=m, group=k, lower=cert['lo'], upper=cert['hi'],
                             practical_loss=result['scores'][m, k], strict_loss=loss,
                             valid=True, active_match=True, reference_in_uniqueness_ball=True))
        lo = np.array([c['lo'] for c in result['certs']]).reshape(3, G).mean(axis=1)
        hi = np.array([c['hi'] for c in result['certs']]).reshape(3, G).mean(axis=1)
        for m in range(3):
            diff = strict['scores'][m] - strict['scores'][2]
            summaries.append(dict(kind=kind, model=m + 1, strict_cv=strict['scores'][m].mean(),
                                  lower=lo[m], upper=hi[m], difference=diff.mean(),
                                  se_proxy=np.std(diff, ddof=1) / np.sqrt(G) if m != 2 else np.nan))
    out = ROOT / 'results/postfit_validation'
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'lexdec_folds.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'lexdec_summary.csv', index=False)
    report = dict(scope='untimed replay of the existing training-only lexdec comparisons',
                  fits=len(rows), comparisons=2, local_tests_pass=300,
                  reference_active_sets_match=True, references_in_uniqueness_balls=True,
                  raw_loss_containment_misses=0, selected_models=[3, 3],
                  max_difference_from_archived_losses=max_loss_difference,
                  original_timing_outputs_unchanged=True)
    (out / 'lexdec_replay.json').write_text(json.dumps(report, indent=2) + '\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
