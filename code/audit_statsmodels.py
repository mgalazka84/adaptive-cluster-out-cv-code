"""Independent REML and BLUP audit for all 300 lexdec held-out fits.

statsmodels uses a single top-level group and two variance-component designs,
as in its official crossed-effects example. Fits use default initial values,
not the prototype's parameters. Predictions use statsmodels' fixed effects
and random-effects estimates, not the prototype prediction kernel.
This is a correctness audit, not a runtime-comparison experiment.
"""
import json
import warnings
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec
from run_lexdec import make_problem

ROOT = Path(__file__).resolve().parents[1]


def external_fit(fold):
    levels = [np.unique(fold.a), np.unique(fold.b)]
    mats = [[np.equal.outer(v, level).astype(float)] for v, level in zip([fold.a, fold.b], levels)]
    vcs = VCSpec(['factor_a', 'factor_b'], [[list(level)] for level in levels], mats)
    model = MixedLM(fold.y, fold.X, groups=np.ones(fold.n),
                    exog_re=np.empty((fold.n, 0)), exog_vc=vcs)
    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter('always')
        result = model.fit(reml=True, method='bfgs', gtol=1e-10, maxiter=500, disp=False)
    theta = np.array([result.scale, *result.vcomp])
    xs, ys, aa, bb = fold.test
    effects = result.random_effects[1.0].to_numpy()
    zsa = np.equal.outer(aa, levels[0]).astype(float)
    zsb = np.equal.outer(bb, levels[1]).astype(float)
    mu = xs@np.asarray(result.fe_params)+zsa@effects[:len(levels[0])]+zsb@effects[len(levels[0]):]
    return theta, mu, result, [str(w.message) for w in warn]


def main():
    out = ROOT/'results'/'lexdec'
    audit = pd.read_csv(out/'fold_audit.csv')
    all_rows = []
    t0 = perf_counter()
    for target in ['new_subject', 'new_word']:
        folds, _, _, groups = make_problem(target)
        for m, fs in enumerate(folds):
            for k, fold in enumerate(fs):
                theta, mu, result, warn = external_fit(fold)
                reference = audit[(audit.target == target) & (audit.model == m) & (audit.group_index == k)].iloc[0]
                expected_theta = reference[[f'reference_theta_{j}' for j in range(3)]].to_numpy(dtype=float)
                loss = float(np.mean((fold.test[1]-mu)**2))
                # Only this diagnostic uses the prototype at external estimates.
                # External losses above are independently computed via BLUPs.
                gradient = float(np.linalg.norm(fold.evaluate(theta).gradient, np.inf))
                row = {'target': target, 'model': m, 'group': str(groups[k]),
                       'group_index': k, 'external_loss': loss,
                       'reference_loss': reference.reference_loss,
                       'loss_difference': loss-reference.reference_loss,
                       'max_variance_difference': float(np.max(np.abs(theta-expected_theta))),
                       'external_converged': bool(result.converged),
                       'external_gradient_in_original_coordinates': gradient,
                       'contained_at_audit_tolerance': reference.lower-1e-8 <= loss <= reference.upper+1e-8,
                       'warnings': ' | '.join(warn)}
                all_rows.append(row)
                if len(all_rows) % 15 == 0:
                    pd.DataFrame(all_rows).to_csv(out/'statsmodels_audit.csv', index=False)
                    print(f'{len(all_rows)}/300 independent fits; current loss difference '
                          f'{row["loss_difference"]:.3g}', flush=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out/'statsmodels_audit.csv', index=False)
    winners = {}
    for target, f in frame.groupby('target'):
        scores = f.groupby('model').external_loss.mean()
        winners[target] = {'scores': scores.to_list(), 'winner': int(scores.idxmin())}
    summary = {'statsmodels_version': statsmodels.__version__, 'fits': len(frame),
               'converged': int(frame.external_converged.sum()),
               'contained': int(frame.contained_at_audit_tolerance.sum()),
               'max_absolute_loss_difference': float(frame.loss_difference.abs().max()),
               'max_absolute_variance_difference': float(frame.max_variance_difference.max()),
               'max_original_coordinate_gradient': float(frame.external_gradient_in_original_coordinates.max()),
               'seconds_for_audit': perf_counter()-t0, 'winners': winners,
               'scope': 'Independent implementation and initialization; still numerical, not a global or rounding-verified reference.',
               'warnings': 'All warning texts are preserved per fold; a software boundary warning is not an exact-zero identification.'}
    (out/'statsmodels_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
