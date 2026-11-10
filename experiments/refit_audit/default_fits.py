"""Check default statsmodels MixedLM fits directly at the returned variances."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec

from cases import HERE, CompressedFold, cases, branch_match
from boundary_acv import active_step, face_certificate, fit_box, training_start


def package_fit(inputs):
    x, y, a, b, (xs, ys, aa, bb) = inputs
    groups = [np.unique(a), np.unique(b)]
    matrices = [[np.equal.outer(v, levels).astype(float)] for v, levels in zip((a,b),groups)]
    vc = VCSpec(['a','b'], [[list(g)] for g in groups], matrices)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        model = MixedLM(y, x, groups=np.ones(len(y)), exog_re=np.empty((len(y),0)), exog_vc=vc)
        result = model.fit()
        theta = np.r_[result.scale, result.vcomp]
        coefficients = result.random_effects[1.0].to_numpy()
        prediction = xs @ np.asarray(result.fe_params)
        offset = 0
        for v, levels in zip((aa,bb),groups):
            prediction += np.equal.outer(v, levels) @ coefficients[offset:offset + len(levels)]
            offset += len(levels)
    return theta, float(np.mean((ys - prediction)**2)), bool(result.converged), [str(w.message) for w in recorded]


def one_call(key, inputs, repeat):
    wall = perf_counter()
    try:
        theta, loss, converged, notes = package_fit(inputs)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        return dict(**key, repeat=repeat, produced=False, error=str(exc))
    package_ms = 1000*(perf_counter()-wall)
    wall = perf_counter()
    fold = CompressedFold(*inputs)
    ev = fold.evaluate(theta)
    cert = face_certificate(fold, theta, ev)
    extra_ms = 1000*(perf_counter()-wall)
    item = dict(**key, repeat=repeat, produced=True, converged=converged, warnings=' | '.join(notes),
                package_ms=package_ms, first_check_ms=extra_ms, first_valid=cert['valid'], reason=cert['reason'],
                mse=loss, residual=theta[0], var_a=theta[1], var_b=theta[2])
    if repeat > 0:
        return item
    # The numerical reference is constructed independently of the external optimizer.
    reference, ref_ev, _ = fit_box(fold, training_start(fold), method='newton')
    reference_loss = float(np.mean((fold.test[1]-fold.prediction(reference,ref_ev)[0])**2))
    at_theta_loss = float(np.mean((fold.test[1]-fold.prediction(theta,ev)[0])**2))
    item.update(reference_loss=reference_loss, prediction_mse_discrepancy=abs(at_theta_loss-loss),
                first_same_root_ball=branch_match(theta,reference,cert),
                first_lower=cert.get('lo',np.nan), first_upper=cert.get('hi',np.nan))
    current = theta.copy()
    steps = 0
    while not cert['valid'] and steps < 20:
        new = active_step(fold,current,ev)
        if np.array_equal(new,current):
            break
        current = new
        ev = fold.evaluate(current)
        cert = face_certificate(fold,current,ev)
        steps += 1
    item.update(repair_steps=steps, after_valid=cert['valid'], after_reason=cert['reason'],
                after_same_root_ball=branch_match(current,reference,cert),
                after_lower=cert.get('lo',np.nan),after_upper=cert.get('hi',np.nan))
    return item


def main(output, mode):
    output.mkdir(exist_ok=True,parents=True)
    timing = mode == 'timing'
    cases_to_use = []
    for name in ['lexdec','orchard']:
        cases_to_use.extend((key,inputs) for key,inputs,selected in cases(name) if not timing or selected)
    if timing:
        for dataset in ['lexdec','orchard']:
            key, inputs = next((k,f) for k,f in cases_to_use if k['dataset']==dataset)
            one_call(key,inputs,1)
    records = []
    rng = np.random.default_rng(14862)
    for repeat in ([1,2,3] if timing else [0]):
        order = rng.permutation(len(cases_to_use)) if timing else range(len(cases_to_use))
        for index in order:
            key, inputs = cases_to_use[index]
            records.append(one_call(key,inputs,repeat))
    frame = pd.DataFrame(records).sort_values(['dataset','target','model_index','fold','repeat'])
    frame.to_csv(output / ('default_timing.csv' if timing else 'default_folds.csv'),index=False)
    if not timing:
        info = {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                'statsmodels':statsmodels.__version__, 'fit_call':'MixedLM.fit()',
                'per_dataset':{}}
        for ds,group in frame.groupby('dataset'):
            done=group[group.produced]
            info['per_dataset'][ds] = dict(attempted=len(group),returned=len(done),
                optimizer_converged=int(done.converged.sum()), first_valid=int(done.first_valid.sum()),
                first_same_ball=int(done.first_same_root_ball.sum()), after_valid=int(done.after_valid.sum()),
                after_same_ball=int(done.after_same_root_ball.sum()), max_steps=int(done.repair_steps.max()),
                max_mse_discrepancy=float(done.prediction_mse_discrepancy.max()),
                reasons=dict(Counter(done.reason)))
        (output/'default_summary.json').write_text(json.dumps(info,indent=2)+'\n')
        print(info, flush=True)
    else:
        med=frame[frame.produced].groupby(['dataset','target','model','fold'])[['package_ms','first_check_ms']].median()
        med['paired_overhead']=med.first_check_ms/med.package_ms
        print(med.groupby('dataset').median(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['accuracy','timing'],default='accuracy')
    parser.add_argument('--output',type=Path,default=HERE/'results')
    args=parser.parse_args()
    main(args.output,args.mode)
