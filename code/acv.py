"""Pilot: local REML validation and cluster-out predictive-loss enclosures.

Dense reference implementation, not a sparse production solver. Analytic
enclosures are evaluated in double precision, without directed rounding.
The root certificate is LOCAL; it does not establish global REML optimality.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import cholesky, solve_triangular, qr, eigh, cho_solve
from scipy.optimize import minimize


FLOOR = 1e-8


def opnorm(a):
    return float(np.linalg.norm(a, 2))


def whiten(l, a):
    v = solve_triangular(l, a, lower=True, check_finite=False)
    return solve_triangular(l, v.T, lower=True, check_finite=False).T


@dataclass
class Evaluation:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray
    b: np.ndarray
    w: np.ndarray
    chol: np.ndarray
    W: object = None
    extra_dim: int = 0
    extra_w2: float = 0.
    extra_b: object = None


class Fold:
    def __init__(self, X, y, a, b, test=None):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.n, self.p = self.X.shape
        if np.linalg.matrix_rank(X) != self.p:
            raise ValueError('Training fixed-effect design is rank deficient')
        self.a, self.b = np.asarray(a), np.asarray(b)
        self.A = np.array([np.eye(self.n),
                           np.equal.outer(a, a).astype(float),
                           np.equal.outer(b, b).astype(float)])
        self.K = qr(self.X, mode='full', check_finite=False)[0][:, self.p:]
        self.C = np.array([self.K.T @ z @ self.K for z in self.A])
        self.z = self.K.T @ self.y
        self.test = test
        self.evaluations = 0

    def evaluate(self, theta):
        self.evaluations += 1
        c = np.einsum('i,ijk->jk', theta, self.C)
        l = cholesky(c, lower=True, check_finite=False)
        w = solve_triangular(l, self.z, lower=True, check_finite=False)
        bs = np.array([whiten(l, a) for a in self.C])
        bw = np.array([b @ w for b in bs])
        value = np.log(np.diag(l)).sum() + 0.5 * (w @ w)
        grad = 0.5 * (np.trace(bs, axis1=1, axis2=2) - bw @ w)
        hess = -0.5 * np.einsum('iab,jab->ij', bs, bs) + bw @ bw.T
        return Evaluation(float(value), grad, (hess+hess.T)/2, bs, w, l)

    def value_gradient(self, theta):
        """Cheaper gradient-only kernel for the L-BFGS-B comparator."""
        self.evaluations += 1
        c = np.einsum('i,ijk->jk', theta, self.C)
        l = cholesky(c, lower=True, check_finite=False)
        inv = cho_solve((l, True), np.eye(len(c)), check_finite=False)
        alpha = inv @ self.z
        val = np.log(np.diag(l)).sum() + .5*(self.z@alpha)
        grad = .5*(np.einsum('ab,iab->i',inv,self.C)-
                    np.einsum('a,iab,b->i',alpha,self.C,alpha))
        return float(val), grad

    def objective(self, theta):
        c = np.einsum('i,ijk->jk', theta, self.C)
        l = cholesky(c, lower=True, check_finite=False)
        w = solve_triangular(l, self.z, lower=True, check_finite=False)
        return float(np.log(np.diag(l)).sum() + 0.5 * (w @ w))

    def prediction(self, theta, ev=None):
        if self.test is None:
            raise ValueError('No held-out observations')
        if ev is None:
            ev = self.evaluate(theta)
        if ev.W is None:
            ev.W = solve_triangular(ev.chol, self.K.T, lower=True,
                                    check_finite=False).T
        xs, ys, aa, bb = self.test
        cross = np.array([np.zeros((len(ys), self.n)),
                          np.equal.outer(aa, self.a).astype(float),
                          np.equal.outer(bb, self.b).astype(float)])
        cst = np.einsum('i,ijk->jk', theta, cross)
        v = np.einsum('i,ijk->jk', theta, self.A)
        l = cholesky(v, lower=True, check_finite=False)
        vinvx = solve_triangular(l.T, solve_triangular(
            l, self.X, lower=True, check_finite=False),
            lower=False, check_finite=False)
        Q = np.linalg.solve(self.X.T @ vinvx, vinvx.T)
        P = ev.W @ ev.W.T
        B = xs @ Q + cst @ P
        mu = B @ self.y
        return mu, B, cross

    def prediction_geometry(self,theta,ev,S):
        mu,B,cross=self.prediction(theta,ev)
        basisV=np.einsum('ij,iab->jab',S,self.A)
        basisC=np.einsum('ij,iab->jab',S,cross)
        G=np.array([(e-B@d)@ev.W for e,d in zip(basisC,basisV)])
        return mu,G


class CompressedFold:
    """Exact Gaussian subspace compression; no observation-space n x n matrix."""
    def __init__(self,X,y,a,b,test=None):
        self.X=np.asarray(X,float);self.y=np.asarray(y,float)
        self.n,self.p=self.X.shape;self.test=test
        if np.linalg.matrix_rank(X)!=self.p:
            raise ValueError('Training design is rank deficient')
        self.a=np.asarray(a);self.b=np.asarray(b)
        self.levela=np.unique(a);self.levelb=np.unique(b)
        Za=np.equal.outer(a,self.levela).astype(float)
        Zb=np.equal.outer(b,self.levelb).astype(float)
        Qx=qr(X,mode='economic',check_finite=False)[0]
        Z=np.column_stack([Za,Zb]);Zres=Z-Qx@(Qx.T@Z)
        Qz,Rz,_=qr(Zres,mode='economic',pivoting=True,check_finite=False)
        diag=np.abs(np.diag(Rz))
        tol=np.finfo(float).eps*max(Zres.shape)*max(1.,diag.max(initial=0))
        rank=int(np.sum(diag>tol));Qz=Qz[:,:rank]
        self.T=np.column_stack([Qx,Qz]);self.r=rank
        self.d0=self.n-self.p-rank
        self.Tx=self.T.T@X;self.ty=self.T.T@y
        self.s0=float(np.linalg.norm(y-self.T@self.ty)**2)
        self.Ua=self.T.T@Za;self.Ub=self.T.T@Zb
        self.A=np.array([np.eye(self.p+rank),self.Ua@self.Ua.T,self.Ub@self.Ub.T])
        self.C=self.A[:,self.p:,self.p:]
        self.z=self.ty[self.p:]
        self.evaluations=0

    def objective(self,theta):
        c=np.einsum('i,ijk->jk',theta,self.C)
        l=cholesky(c,lower=True,check_finite=False)
        w=solve_triangular(l,self.z,lower=True,check_finite=False)
        return float(np.log(np.diag(l)).sum()+.5*w@w+
                     .5*self.d0*np.log(theta[0])+.5*self.s0/theta[0])

    def evaluate(self,theta):
        self.evaluations+=1
        c=np.einsum('i,ijk->jk',theta,self.C)
        l=cholesky(c,lower=True,check_finite=False)
        w=solve_triangular(l,self.z,lower=True,check_finite=False)
        bs=np.array([whiten(l,a) for a in self.C])
        bw=np.array([b@w for b in bs])
        val=np.log(np.diag(l)).sum()+.5*w@w+.5*self.d0*np.log(theta[0])+.5*self.s0/theta[0]
        grad=.5*(np.trace(bs,axis1=1,axis2=2)-bw@w)
        grad[0]+=.5*self.d0/theta[0]-.5*self.s0/theta[0]**2
        H=-.5*np.einsum('iab,jab->ij',bs,bs)+bw@bw.T
        H[0,0]+=-.5*self.d0/theta[0]**2+self.s0/theta[0]**3
        return Evaluation(float(val),grad,(H+H.T)/2,bs,w,l,
            extra_dim=self.d0,extra_w2=self.s0/theta[0],
            extra_b=np.array([1/theta[0],0.,0.]))

    def value_gradient(self,theta):
        self.evaluations+=1
        c=np.einsum('i,ijk->jk',theta,self.C)
        l=cholesky(c,lower=True,check_finite=False)
        inv=cho_solve((l,True),np.eye(len(c)),check_finite=False)
        alpha=inv@self.z
        val=np.log(np.diag(l)).sum()+.5*self.z@alpha+.5*self.d0*np.log(theta[0])+.5*self.s0/theta[0]
        g=.5*(np.einsum('ab,iab->i',inv,self.C)-np.einsum('a,iab,b->i',alpha,self.C,alpha))
        g[0]+=.5*self.d0/theta[0]-.5*self.s0/theta[0]**2
        return float(val),g

    def prediction(self,theta,ev=None):
        if ev is None:ev=self.evaluate(theta)
        xs,ys,aa,bb=self.test
        Zsa=np.equal.outer(aa,self.levela).astype(float)
        Zsb=np.equal.outer(bb,self.levelb).astype(float)
        cross=np.array([np.zeros((len(ys),self.p+self.r)),Zsa@self.Ua.T,Zsb@self.Ub.T])
        cst=np.einsum('i,ijk->jk',theta,cross)
        v=np.einsum('i,ijk->jk',theta,self.A)
        l=cholesky(v,lower=True,check_finite=False)
        vinvx=cho_solve((l,True),self.Tx,check_finite=False)
        Q=np.linalg.solve(self.Tx.T@vinvx,vinvx.T)
        P=np.zeros_like(v)
        P[self.p:,self.p:]=cho_solve((ev.chol,True),np.eye(self.r),check_finite=False)
        B=xs@Q+cst@P
        return B@self.ty,B,cross

    def prediction_geometry(self,theta,ev,S):
        mu,B,cross=self.prediction(theta,ev)
        basisV=np.einsum('ij,iab->jab',S,self.A)
        basisC=np.einsum('ij,iab->jab',S,cross)
        W=np.zeros((self.p+self.r,self.r))
        W[self.p:]=solve_triangular(ev.chol,np.eye(self.r),lower=True,check_finite=False).T
        G=np.array([(e-B@d)@W for e,d in zip(basisC,basisV)])
        return mu,G


def fit(fold, start, gtol=1e-7):
    """Warm-started numerical reference; local box-constrained REML fit."""
    before = fold.evaluations
    def fun(theta):
        return fold.value_gradient(theta)
    opt = minimize(fun, np.maximum(start, FLOOR), jac=True, method='L-BFGS-B',
                   bounds=[(FLOOR, None)]*3,
                   options={'ftol': 1e-13, 'gtol': gtol, 'maxiter': 400,
                            'maxls': 40})
    theta = opt.x.copy()
    # Tighten interior numerical roots independently of the CV intervals.
    for _ in range(8):
        ev = fold.evaluate(theta)
        pg = ev.gradient.copy()
        pg[(theta <= FLOOR*1.01) & (pg > 0)] = 0
        if np.linalg.norm(pg, np.inf) < gtol:
            break
        if theta.min() <= FLOOR*1.01 or np.linalg.eigvalsh(ev.hessian).min() <= 0:
            break
        proposal = newton_step(fold, theta, ev)
        if np.linalg.norm(proposal-theta) < 1e-14:
            break
        theta = proposal
    ev = fold.evaluate(theta)
    pg = ev.gradient.copy()
    pg[(theta <= FLOOR*1.01) & (pg > 0)] = 0
    return theta, ev, {'n_eval': fold.evaluations-before,
                       'optimizer_success': bool(opt.success),
                       'projected_gradient': float(np.linalg.norm(pg, np.inf)),
                       'boundary': bool(theta.min() <= FLOOR*1.01)}


def fit_newton(fold, start, gtol=1e-9):
    """Strong comparator: the same damped Newton kernel, run to convergence."""
    before = fold.evaluations
    theta = np.maximum(start.copy(), FLOOR)
    for _ in range(30):
        ev = fold.evaluate(theta)
        pg = ev.gradient.copy()
        pg[(theta <= FLOOR*1.01) & (pg > 0)] = 0
        if np.linalg.norm(pg, np.inf) < gtol:
            return theta, ev, {'n_eval':fold.evaluations-before,
                'projected_gradient':float(np.linalg.norm(pg,np.inf)),
                'boundary':bool(theta.min()<=FLOOR*1.01), 'fallback':False}
        if theta.min() <= FLOOR*1.01 or np.linalg.eigvalsh(ev.hessian).min() <= 0:
            break
        proposal = newton_step(fold,theta,ev)
        if np.linalg.norm(proposal-theta) < 1e-14:
            break
        theta=proposal
    theta,ev,meta=fit(fold,theta,gtol=gtol)
    meta['n_eval']=fold.evaluations-before
    meta['fallback']=True
    return theta,ev,meta


def newton_step(fold, theta, ev=None):
    if ev is None:
        ev = fold.evaluate(theta)
    vals, vecs = eigh(ev.hessian, check_finite=False)
    vals = np.maximum(vals, max(0.01, float(np.max(np.abs(vals)))*1e-5))
    step = -(vecs / vals) @ (vecs.T @ ev.gradient)
    negative = step < 0
    rate = min(1., float(np.min(0.99*(theta[negative]-FLOOR)/
                               -step[negative])) if negative.any() else 1.)
    slope = ev.gradient @ step
    for _ in range(30):
        proposal = np.maximum(theta + rate*step, FLOOR)
        if fold.objective(proposal) <= ev.value + 1e-4*rate*slope + 1e-12:
            return proposal
        rate *= 0.5
    return theta.copy()


def certificate(fold, theta, ev=None):
    """Compute a local stationary-root and held-out MSE enclosure.

    Symmetric covariance-direction norms use eigenvalues; prediction-map norms
    use Frobenius upper bounds. No operation uses directed rounding.
    """
    if ev is None:
        ev = fold.evaluate(theta)
    lam, U = eigh(ev.hessian, check_finite=False)
    if lam[0] <= 1e-10:
        return {'valid': False, 'reason': 'nonpositive_curvature'}
    S = (U / np.sqrt(lam)) @ U.T
    eta = float(np.linalg.norm(S @ ev.gradient))
    r = max(2*eta, 1e-12)
    if np.any(theta - r*np.linalg.norm(S, axis=1) <= 0):
        return {'valid': False, 'reason': 'variance_boundary', 'eta': eta}
    # A symmetric basis with sum_i S_ij * derivative_i.
    D = np.einsum('ij,iab->jab', S, ev.b)
    # Frobenius bounds are rigorous upper bounds in exact arithmetic and cheap.
    extra=np.zeros(3) if ev.extra_b is None else ev.extra_b@S
    f = float(np.sqrt(np.linalg.norm(D)**2+ev.extra_dim*np.linalg.norm(extra)**2))
    # Tight spectral norms for the small dense proof of concept.
    ops=np.array([np.max(np.abs(eigh(d,eigvals_only=True,check_finite=False))) for d in D])
    a = float(np.linalg.norm(np.maximum(ops,np.abs(extra))))
    if a*r >= 1:
        return {'valid': False, 'reason': 'covariance_radius', 'eta': eta}
    Dw = np.array([d @ ev.w for d in D]).T
    b0 = np.sqrt(max(0, np.linalg.eigvalsh(Dw.T @ Dw+ev.extra_w2*np.outer(extra,extra))[-1]))
    wnorm=np.sqrt(ev.w@ev.w+ev.extra_w2)
    b = b0 + a*a*r/(1-a*r)*wnorm
    L = f*f*a/(1-a*r)**3 + 3*b*b*a/(1-a*r)**2
    if L*r >= 1:
        return {'valid': False, 'reason': 'contraction', 'eta': eta,
                'Lr': float(L*r)}
    # r=2 eta and Lr<1 imply self mapping and contraction.
    disc = max(0., 1-2*L*eta)
    rho = 2*eta/(1+np.sqrt(disc)) if eta else 0.
    mu,G=fold.prediction_geometry(theta,ev,S)
    Gw = np.array([g @ ev.w for g in G]).T
    g0 = np.sqrt(max(0, np.linalg.eigvalsh(Gw.T @ Gw)[-1]))
    g = float(np.linalg.norm(G))  # Frobenius upper bound
    a_pred=float(np.linalg.norm(ops))
    pred_radius = rho*g0 + g*a_pred*rho*rho*np.linalg.norm(ev.w)/(1-a_pred*rho)
    residual = np.linalg.norm(fold.test[1]-mu)
    ns = len(mu)
    lo = max(0., residual-pred_radius)**2/ns
    hi = (residual+pred_radius)**2/ns
    return {'valid': True, 'reason': 'local_root', 'lo': float(lo),
            'hi': float(hi), 'mse': float(residual**2/ns),
            'eta': eta, 'radius': float(rho), 'outer_radius': float(r),
            'Lr': float(L*r), 'prediction_radius': float(pred_radius),
            'S': S, 'theta': theta.copy()}


def simulate(seed, regime='balanced', na=12, nb=10):
    rng = np.random.default_rng(seed)
    aa, bb = np.meshgrid(np.arange(na), np.arange(nb), indexing='ij')
    aa, bb = np.repeat(aa.ravel(),2), np.repeat(bb.ravel(),2)
    if regime == 'sparse':
        keep = rng.random(len(aa)) < 0.55
        aa, bb = aa[keep], bb[keep]
    if regime == 'unbalanced':
        keep = rng.random(len(aa)) < np.where(aa < 3, .95, .35)
        aa, bb = aa[keep], bb[keep]
    n = len(aa)
    x1 = rng.normal(size=n)
    x2 = .65*x1 + np.sqrt(1-.65**2)*rng.normal(size=n)
    x3 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2, x3])
    signal = 0.12 if regime == 'near_tie' else .75
    va = .0001 if regime == 'boundary' else .8
    vb = .55
    y = 1+1.2*x1+signal*x2+rng.normal(scale=np.sqrt(va),size=na)[aa]
    y += rng.normal(scale=np.sqrt(vb),size=nb)[bb]+rng.normal(size=n)
    return X, y, aa, bb


def make_fold(X,y,a,b,columns,group,implementation='dense'):
    tr = a != group
    te = ~tr
    cls=Fold if implementation=='dense' else CompressedFold
    return cls(X[tr][:,columns], y[tr], a[tr], b[tr],
                (X[te][:,columns],y[te],a[te],b[te]))
