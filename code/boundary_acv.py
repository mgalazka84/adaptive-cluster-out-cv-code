"""Active-face extension of the local REML/loss enclosure.

Random-effect variances may be exactly zero. A reduced-Hessian root enclosure
is accepted only when all active one-sided derivatives have a positive lower
bound. This is a LOCAL, exact-arithmetic argument evaluated in ordinary float64.
It neither verifies rounding nor identifies a global REML optimum.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from acv import CompressedFold

LOWER = np.array([1e-10, 0., 0.])


def projected_gradient(theta, gradient):
    pg = gradient.copy()
    pg[(theta <= LOWER) & (pg > 0)] = 0.
    return pg


def active_step(fold, theta, ev=None):
    """Projected, damped Newton step; can both enter and leave a zero face."""
    if ev is None:
        ev = fold.evaluate(theta)
    fixed = (theta <= LOWER) & (ev.gradient >= 0)
    free = np.flatnonzero(~fixed)
    if len(free) == 0:
        return theta.copy()
    H = ev.hessian[np.ix_(free, free)]
    lam, U = eigh(H, check_finite=False)
    lam = np.maximum(lam, max(.01, np.max(np.abs(lam))*1e-5))
    d = np.zeros(3)
    d[free] = -(U/lam) @ (U.T @ ev.gradient[free])
    displacement = np.maximum(theta+d, LOWER)-theta
    if ev.gradient @ displacement >= 0:
        d = -ev.gradient/np.maximum(np.abs(np.diag(ev.hessian)), .01)
        d[fixed] = 0
    rate = 1.
    for _ in range(40):
        proposal = np.maximum(theta+rate*d, LOWER)
        delta = proposal-theta
        slope = float(ev.gradient @ delta)
        if slope <= 0 and fold.objective(proposal) <= ev.value+1e-4*slope+1e-12:
            return proposal
        rate *= .5
    return theta.copy()


def fit_box(fold, start, gtol=1e-9, method='lbfgs', ftol=1e-14, polish=True):
    """Numerical reference with exact nonnegative random-effect variances."""
    before = fold.evaluations
    theta = np.maximum(np.asarray(start, float), LOWER)
    optimizer_success = True
    newton_iterations = 0
    lbfgs_iterations = 0
    used_lbfgs = method == 'lbfgs'
    if method == 'newton':
        for _ in range(60):
            ev = fold.evaluate(theta)
            if np.linalg.norm(projected_gradient(theta, ev.gradient), np.inf) < gtol:
                break
            proposal = active_step(fold, theta, ev)
            if np.max(np.abs(proposal-theta)) < 1e-15:
                break
            theta = proposal
            newton_iterations += 1
        ev = fold.evaluate(theta)
        used_lbfgs = np.linalg.norm(projected_gradient(theta, ev.gradient), np.inf) >= gtol
    if used_lbfgs:
        # Rescale all covariance coordinates by a training-only response scale.
        scale = max(float(np.var(fold.y)), 1e-5)
        def fun(z):
            value, gradient = fold.value_gradient(z*scale)
            return value, gradient*scale
        opt = minimize(fun, theta/scale, jac=True, method='L-BFGS-B',
                       bounds=[(x/scale, None) for x in LOWER],
                       options={'ftol': ftol, 'gtol': gtol*scale,
                                'maxiter': 500, 'maxls': 40})
        theta = np.maximum(opt.x*scale, LOWER)
        optimizer_success = bool(opt.success)
        lbfgs_iterations = int(opt.nit)
    # Polish free coordinates even when a random variance is zero.
    for _ in range(12 if polish else 0):
        ev = fold.evaluate(theta)
        if np.linalg.norm(projected_gradient(theta, ev.gradient), np.inf) < gtol:
            break
        proposal = active_step(fold, theta, ev)
        if np.max(np.abs(proposal-theta)) < 1e-15:
            break
        theta = proposal
        newton_iterations += 1
    ev = fold.evaluate(theta)
    return theta, ev, {
        'n_eval': fold.evaluations-before,
        'iterations': newton_iterations+lbfgs_iterations,
        'newton_iterations': newton_iterations, 'lbfgs_iterations': lbfgs_iterations,
        'projected_gradient': float(np.linalg.norm(projected_gradient(theta, ev.gradient), np.inf)),
        'optimizer_success': optimizer_success, 'used_lbfgs': used_lbfgs,
        'active': np.flatnonzero(theta == 0).tolist(),
        'residual_guard': bool(theta[0] <= LOWER[0])}


def active_gradient_lower(ev, index, t):
    """PSD directional derivative lower bound throughout ||u|| <= rho.

Here t = a*rho < 1. B_index is positive semidefinite by construction. The
compressed scalar block is included in trace, quadratic, and vector norms.
"""
    B = ev.b[index]
    extra = 0. if ev.extra_b is None or ev.extra_dim == 0 else ev.extra_b[index]
    trace = float(np.trace(B)+ev.extra_dim*extra)
    q = max(0., float(ev.w @ B @ ev.w+extra*ev.extra_w2))
    beta = max(float(eigh(B, eigvals_only=True, check_finite=False)[-1]), extra)
    wnorm = np.sqrt(ev.w@ev.w+ev.extra_w2)
    delta = t/(1-t)
    # First bound uses trace monotonicity and the B^(1/2) seminorm.
    bound1 = .5*(trace/(1+t)-(np.sqrt(q)+np.sqrt(max(beta, 0))*delta*wnorm)**2)
    # Second is the perturbation expansion about the observed gradient.
    Bw = B@ev.w
    bwnorm = np.sqrt(Bw@Bw+extra*extra*ev.extra_w2)
    bound2 = ev.gradient[index]-.5*(delta*trace+2*bwnorm*delta*wnorm+
                                   beta*delta*delta*wnorm*wnorm)
    return float(max(bound1, bound2))


def face_certificate(fold, theta, ev=None, allow_boundary=True):
    if ev is None:
        ev = fold.evaluate(theta)
    if theta[0] <= 0 or np.any(theta[1:] < 0):
        return {'valid': False, 'reason': 'infeasible'}
    active = np.flatnonzero(theta == 0)
    if len(active) and not allow_boundary:
        return {'valid': False, 'reason': 'boundary_disabled'}
    free = np.flatnonzero(theta > 0)
    lam, U = eigh(ev.hessian[np.ix_(free, free)], check_finite=False)
    if lam[0] <= 1e-10:
        return {'valid': False, 'reason': 'nonpositive_face_curvature'}
    S = np.zeros((3, len(free)))
    S[free] = (U/np.sqrt(lam))@U.T
    eta = float(np.linalg.norm(S.T@ev.gradient))
    r = max(2*eta, 1e-12)
    if np.any(theta[free]-r*np.linalg.norm(S[free], axis=1) <= 0):
        return {'valid': False, 'reason': 'free_variance_boundary', 'eta': eta}
    D = np.einsum('ij,iab->jab', S, ev.b)
    extra = (np.zeros(len(free)) if ev.extra_b is None or ev.extra_dim == 0
             else ev.extra_b@S)
    f2 = float(np.linalg.norm(D)**2+ev.extra_dim*(extra@extra))
    ops = np.array([np.max(np.abs(eigh(d, eigvals_only=True, check_finite=False))) for d in D])
    a = float(np.linalg.norm(np.maximum(ops, np.abs(extra))))
    if a*r >= 1:
        return {'valid': False, 'reason': 'covariance_radius', 'eta': eta}
    Dw = np.array([d@ev.w for d in D]).T
    b0 = np.sqrt(max(0., np.linalg.eigvalsh(Dw.T@Dw+ev.extra_w2*np.outer(extra, extra))[-1]))
    wnorm = np.sqrt(ev.w@ev.w+ev.extra_w2)
    b = b0+a*a*r/(1-a*r)*wnorm
    L = f2*a/(1-a*r)**3+3*b*b*a/(1-a*r)**2
    if L*r >= 1:
        return {'valid': False, 'reason': 'contraction', 'eta': eta, 'Lr': float(L*r)}
    rho = 2*eta/(1+np.sqrt(max(0., 1-2*L*eta))) if eta else 0.
    lower = np.array([active_gradient_lower(ev, j, a*rho) for j in active])
    if np.any(lower <= 0):
        return {'valid': False, 'reason': 'active_gradient_not_separated',
                'active_lower': lower, 'active': active, 'eta': eta}
    mu, G = fold.prediction_geometry(theta, ev, S)
    Gw = np.array([g@ev.w for g in G]).T
    c0 = np.sqrt(max(0., np.linalg.eigvalsh(Gw.T@Gw)[-1]))
    c = float(np.linalg.norm(G))
    a_pred = float(np.linalg.norm(ops))
    radius = rho*c0+c*a_pred*rho*rho*np.linalg.norm(ev.w)/(1-a_pred*rho)
    residual = np.linalg.norm(fold.test[1]-mu)
    ns = len(mu)
    return {'valid': True, 'reason': 'active_face' if len(active) else 'interior',
            'lo': float(max(0., residual-radius)**2/ns),
            'hi': float((residual+radius)**2/ns), 'mse': float(residual**2/ns),
            'raw_lo': float(max(0., residual-radius)**2/ns),
            'raw_hi': float((residual+radius)**2/ns),
            'eta': eta, 'radius': float(rho), 'outer_radius': float(r),
            'Lr': float(L*r), 'a': a, 'prediction_radius': float(radius),
            'f2': f2, 'b0': float(b0), 'b_r': float(b), 'c': c, 'c0': float(c0),
            'trace_L': float(f2*a/(1-a*r)**3),
            'quadratic_L': float(3*b*b*a/(1-a*r)**2),
            'active': active, 'free': free, 'active_lower': lower,
            'S': S, 'theta': theta.copy()}


def boundary_tube(ev, cert):
    """Explicit relative neighbourhood with no other KKT point.

    Free coordinates satisfy ||u|| <= R; each original active variance is
    between zero and epsilon. Computed after validation, not in the scheduler.
    The proof uses the same two derivative bounds for a general symmetric
    covariance perturbation, not only for perturbations within the face.
    """
    active = cert.get('active', [])
    if not cert.get('valid') or not len(active):
        return None
    t0 = cert['a']*cert['radius']
    lo, hi = t0, 1.-1e-10
    for _ in range(60):
        mid = (lo+hi)/2
        if min(active_gradient_lower(ev, j, mid) for j in active) > 0:
            lo = mid
        else:
            hi = mid
    tdag = (t0+lo)/2
    slack = tdag-t0
    if slack <= 0:
        return None
    R = cert['radius']+min((cert['outer_radius']-cert['radius'])/2,
                           slack/(2*max(cert['a'], 1e-300)))
    beta_sum = 0.
    for j in active:
        extra = 0. if ev.extra_b is None or ev.extra_dim == 0 else ev.extra_b[j]
        beta_sum += max(eigh(ev.b[j], eigvals_only=True, check_finite=False)[-1], extra)
    epsilon = (tdag-cert['a']*R)/beta_sum
    return {'free_radius': float(R), 'active_epsilon': float(epsilon),
            'perturbation_radius': float(tdag),
            'minimum_derivative': float(min(active_gradient_lower(ev,j,tdag) for j in active))}


def same_root(old, new):
    """Sufficient containment condition for successive roots on the same face."""
    if not np.array_equal(old['active'], new['active']):
        return False
    free = old['free']
    shift = np.linalg.solve(old['S'][free], (new['theta']-old['theta'])[free])
    transfer = np.linalg.solve(old['S'][free], new['S'][free])
    return np.linalg.norm(shift)+new['radius']*np.linalg.norm(transfer, 2) < old['outer_radius']


@dataclass
class State:
    fold: CompressedFold
    theta: np.ndarray
    model: int
    group: int
    allow_boundary: bool = True
    steps: int = 0
    checks: int = 0
    numerical: bool = False
    ev: object = None
    cert: object = None
    fallback_reason: str = ''
    fallback_iterations: int = 0
    anchor: object = None

    def refine(self):
        if self.numerical:
            return
        if self.ev is None:
            self.ev = self.fold.evaluate(self.theta)
        if self.steps >= 7:
            self.complete('refinement_budget')
            return
        # Retain the last valid enclosure across unsuccessful intermediate tests.
        old = self.anchor
        self.theta = active_step(self.fold, self.theta, self.ev)
        self.steps += 1
        self.ev = self.fold.evaluate(self.theta)
        self.cert = face_certificate(self.fold, self.theta, self.ev, self.allow_boundary)
        self.checks += 1
        if old and self.cert['valid'] and not same_root(old, self.cert):
            self.complete('root_anchor')
            return
        if old and self.cert['valid']:
            lo = max(old['lo'], self.cert['lo'])
            hi = min(old['hi'], self.cert['hi'])
            if lo > hi:
                self.complete('empty_interval_intersection')
                return
            self.cert['lo'], self.cert['hi'] = lo, hi
        if self.cert['valid']:
            self.anchor = self.cert
        if not self.cert['valid'] and self.steps >= 4:
            self.complete(self.cert['reason'])

    def complete(self, reason):
        self.theta, self.ev, meta = fit_box(self.fold, self.theta, method='newton')
        self.fallback_iterations += meta['iterations']
        loss = float(np.mean((self.fold.test[1]-self.fold.prediction(self.theta, self.ev)[0])**2))
        self.cert = {'valid': False, 'reason': 'numerical_refit', 'lo': loss, 'hi': loss, 'mse': loss}
        self.numerical = True
        self.fallback_reason = reason


def adaptive(folds, starts, allow_boundary=True,
             selection_atol=1e-10, selection_rtol=1e-10):
    """Refine until loss intervals separate beyond a numerical tie tolerance.

    The tolerance is a conservative decision guard, not a bound on all
    floating-point or optimization error. Zero tolerances recover the original
    strict-comparison rule used for the archived version-2 timings.
    """
    if (not np.isfinite(selection_atol) or not np.isfinite(selection_rtol)
            or selection_atol < 0 or selection_rtol < 0):
        raise ValueError('Selection tolerances must be finite and nonnegative')
    states = []
    for m, fs in enumerate(folds):
        for k, fold in enumerate(fs):
            s = State(fold, starts[m].copy(), m, k, allow_boundary)
            while s.cert is None or 'lo' not in s.cert:
                s.refine()
            states.append(s)
    rounds = 0
    while True:
        lower = np.array([np.mean([s.cert['lo'] for s in states if s.model == m]) for m in range(len(folds))])
        upper = np.array([np.mean([s.cert['hi'] for s in states if s.model == m]) for m in range(len(folds))])
        best = int(np.argmin(upper))
        other_lower = np.min(np.delete(lower, best))
        margin = selection_atol+selection_rtol*max(abs(upper[best]), abs(other_lower))
        if upper[best]+margin < other_lower:
            return best, states, lower, upper, rounds
        active = np.flatnonzero(lower <= upper[best]+margin)
        choices = [s for s in states if s.model in active and not s.numerical]
        if not choices:
            return None, states, lower, upper, rounds
        s = max(choices, key=lambda s: s.cert['hi']-s.cert['lo'])
        s.refine()
        while 'lo' not in s.cert:
            s.refine()
        rounds += 1
        if rounds > 10000:
            raise RuntimeError('Unexpected refinement loop')
