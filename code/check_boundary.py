"""Independent numerical checks of active-face statements (not proofs)."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from acv import simulate, make_fold, CompressedFold
from boundary_acv import (fit_box, face_certificate, active_step,
                          active_gradient_lower, adaptive, boundary_tube)


def main():
    rng = np.random.default_rng(5701)
    counts = {'face_enclosures': 0, 'double_active': 0, 'gradient_ball_checks': 0,
              'loss_ball_checks': 0, 'dense_compressed': 0, 'wrong_faces_rejected': 0,
              'full_neighbourhood_checks': 0}
    max_compression = 0.
    max_kkt_violation = 0.
    for seed in range(30):
        X, y, a, b = simulate(12000+seed, 'boundary', na=7, nb=6)
        if seed >= 15:
            # Independent no-random-effects data, preserving the fixed design.
            y = 1+X[:, 1]+.4*X[:, 2]+rng.normal(size=len(y))
        f = make_fold(X, y, a, b, [0, 1, 2], seed % 7, 'compressed')
        fd = make_fold(X, y, a, b, [0, 1, 2], seed % 7, 'dense')
        ref, er, _ = fit_box(f, [1, .2, .2])
        active = np.flatnonzero(ref == 0)
        if len(active) == 0:
            continue
        center = ref.copy()
        center[center > 0] *= 1.00001  # Nonzero-radius check, not just exact roots.
        ev = f.evaluate(center)
        c = face_certificate(f, center, ev)
        assert c['valid'], (seed, c)
        norm = face_certificate(f, center, ev, loss_bound='norm')
        assert c['lo'] >= norm['lo'] and c['hi'] <= norm['hi']
        assert c['radius'] > 1e-8
        free = c['free']
        distance = np.linalg.norm(np.linalg.solve(c['S'][free], (ref-center)[free]))
        assert distance <= c['radius']+1e-7
        loss = np.mean((f.test[1]-f.prediction(ref, er)[0])**2)
        assert c['lo']-1e-9 <= loss <= c['hi']+1e-9
        counts['face_enclosures'] += 1
        counts['double_active'] += int(len(active) == 2)
        ed = fd.evaluate(center)
        cd = face_certificate(fd, center, ed)
        assert cd['valid']
        max_compression = max(max_compression,
            abs(ev.value-ed.value), np.max(np.abs(ev.gradient-ed.gradient)),
            np.max(np.abs(ev.hessian-ed.hessian)),
            np.max(np.abs(f.prediction(center, ev)[0]-fd.prediction(center, ed)[0])))
        counts['dense_compressed'] += 1
        tube=boundary_tube(ev,c)
        assert tube and tube['active_epsilon']>0 and tube['free_radius']>c['radius']
        for _ in range(20):
            u=rng.normal(size=len(free));u*=tube['free_radius']/np.linalg.norm(u)
            t=center+c['S']@u
            t[active]=rng.uniform(0,tube['active_epsilon'],size=len(active))
            ef=f.evaluate(t)
            assert min(ef.gradient[active])>=tube['minimum_derivative']-1e-8
            counts['full_neighbourhood_checks']+=1
        for _ in range(20):
            u = rng.normal(size=len(free)); u *= c['radius']/np.linalg.norm(u)
            t = center+c['S']@u
            e2 = f.evaluate(t)
            for j, lower in zip(c['active'], c['active_lower']):
                max_kkt_violation = max(max_kkt_violation, lower-e2.gradient[j])
                assert lower <= e2.gradient[j]+1e-9
                counts['gradient_ball_checks'] += 1
            l2 = np.mean((f.test[1]-f.prediction(t, e2)[0])**2)
            assert c['lo']-1e-9 <= l2 <= c['hi']+1e-9
            counts['loss_ball_checks'] += 1
    # Face stationarity alone is insufficient: construct wrong-face optima.
    for seed in range(10):
        X, y, a, b = simulate(4700+seed, 'balanced', na=7, nb=6)
        f = make_fold(X, y, a, b, [0, 1, 2], 0, 'compressed')
        def fun(z):
            t = np.array([z[0], 0., z[1]])
            v, g = f.value_gradient(t)
            return v, g[[0, 2]]
        opt = minimize(fun, [1., .5], jac=True, method='L-BFGS-B',
                       bounds=[(1e-10, None), (1e-10, None)],
                       options={'ftol': 1e-14, 'gtol': 1e-10})
        t = np.array([opt.x[0], 0., opt.x[1]])
        e = f.evaluate(t)
        if e.gradient[1] < -1e-5:
            cert = face_certificate(f, t, e)
            assert not cert['valid'], cert
            released = active_step(f, t, e)
            assert released[1] > 0
            counts['wrong_faces_rejected'] += 1
    # An exactly repeated candidate remains an unresolved numerical tie.
    X, y, a, b = simulate(991, 'boundary', na=5, nb=4)
    theta, _, _ = fit_box(CompressedFold(X[:, :3], y, a, b), [1., .3, .3])
    folds = [make_fold(X, y, a, b, [0, 1, 2], k, 'compressed') for k in np.unique(a)]
    winner, _, lower, upper, _ = adaptive([folds, folds], [theta, theta])
    assert winner is None, (winner, lower, upper)
    # Independent objects and slightly different starts must not turn rounding
    # differences between the same statistical model into a unique winner.
    winner, _, lower, upper, _ = adaptive(
        [folds, deepcopy(folds)], [theta, theta*(1+1e-12)])
    assert winner is None, (winner, lower, upper)
    assert counts['face_enclosures'] >= 10 and counts['double_active'] > 0
    assert counts['wrong_faces_rejected'] > 0
    assert max_compression < 1e-8
    # The practical fit must never request a Hessian, including at termination.
    f=folds[0]
    full_theta,full_ev,_=fit_box(f,theta,gtol=1e-6,ftol=1e-9,polish=False)
    original=f.evaluate
    def forbidden(*args,**kwargs):raise AssertionError('Unnecessary Hessian')
    f.evaluate=forbidden
    light_theta,light_ev,_=fit_box(f,theta,gtol=1e-6,ftol=1e-9,polish=False,terminal_hessian=False)
    assert np.array_equal(full_theta,light_theta)
    assert np.allclose(f.prediction(light_theta,light_ev)[0],f.prediction(full_theta,full_ev)[0],atol=1e-13,rtol=1e-13)
    f.evaluate=original
    result = {'status': 'passed', **counts, 'identical_model_tie_checks': 2,
              'gradient_only_terminal_check':True,'directional_subset_checks':counts['face_enclosures'],
              'max_dense_compressed_difference': float(max_compression),
              'max_sampled_active_bound_violation': float(max_kkt_violation),
              'scope': 'Numerical checks, not directed-rounding verification or a theorem proof.'}
    path = Path(__file__).resolve().parents[1]/'results'/'boundary_checks.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
