"""Reproducible simulation; reference fits are never used by the adaptive arm."""
import argparse, json, platform, sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
from acv import (Fold, CompressedFold, simulate, make_fold, fit, fit_newton, newton_step,
                 certificate)


@dataclass
class State:
    fold: Fold
    theta: np.ndarray
    model: int
    group: int
    steps: int = 0
    checks: int = 0
    numerical: bool = False
    ev: object = None
    cert: object = None

    def refine(self):
        if self.numerical:
            return
        if self.ev is None:
            self.ev = self.fold.evaluate(self.theta)
        if self.steps >= 5:
            self.complete()
            return
        old_cert=self.cert if self.cert and self.cert.get('valid') else None
        self.theta = newton_step(self.fold, self.theta, self.ev)
        self.steps += 1
        self.ev = self.fold.evaluate(self.theta)
        self.cert = certificate(self.fold, self.theta, self.ev)
        self.checks += 1
        if old_cert and self.cert['valid']:
            # Ensure the refined root belongs to the preceding uniqueness ball.
            shift=np.linalg.solve(old_cert['S'],self.theta-old_cert['theta'])
            transfer=np.linalg.solve(old_cert['S'],self.cert['S'])
            if np.linalg.norm(shift)+self.cert['radius']*np.linalg.norm(transfer,2) >= old_cert['outer_radius']:
                self.complete()
                return
        if not self.cert['valid'] and self.steps >= 3:
            self.complete()

    def complete(self):
        self.theta, self.ev, _ = fit_newton(self.fold, self.theta)
        mu,_,_ = self.fold.prediction(self.theta,self.ev)
        loss = float(np.mean((self.fold.test[1]-mu)**2))
        self.cert = {'valid':False,'reason':'numerical_refit',
                     'lo':loss,'hi':loss,'mse':loss}
        self.numerical = True


def adaptive(folds, starts, ng):
    states=[]
    for m, fs in enumerate(folds):
        for k, fold in enumerate(fs):
            s=State(fold,starts[m].copy(),m,k)
            while s.cert is None or ('lo' not in s.cert):
                s.refine()
            states.append(s)
    rounds=0
    while True:
        lower=np.array([np.mean([s.cert['lo'] for s in states if s.model==m])
                        for m in range(len(folds))])
        upper=np.array([np.mean([s.cert['hi'] for s in states if s.model==m])
                        for m in range(len(folds))])
        best=int(np.argmin(upper))
        others=np.delete(lower,best)
        if upper[best] < np.min(others):
            break
        active=np.flatnonzero(lower <= upper[best])
        choices=[s for s in states if s.model in active and not s.numerical]
        if not choices:
            # Do not turn a numerical tie into an asserted unique winner.
            return None,states,lower,upper,rounds
        s=max(choices,key=lambda s:s.cert['hi']-s.cert['lo'])
        s.refine()
        while 'lo' not in s.cert:
            s.refine()
        rounds+=1
        if rounds > 10000:
            raise RuntimeError('Unexpected refinement loop')
    return best,states,lower,upper,rounds


def trial(regime,seed,na,nb,implementation='dense'):
    X,y,a,b=simulate(seed,regime,na=na,nb=nb)
    groups=np.unique(a)
    columns=[[0,1],[0,1,2],[0,1,2,3]]
    common_start=perf_counter()
    starts=[]
    for cols in columns:
        cls=Fold if implementation=='dense' else CompressedFold
        full=cls(X[:,cols],y,a,b)
        theta,_,_=fit(full,np.array([1.,.7,.5]),gtol=1e-9)
        starts.append(theta)
    folds=[[make_fold(X,y,a,b,cols,k,implementation) for k in groups] for cols in columns]
    common_seconds=perf_counter()-common_start
    reference={}; times={}; evals={}; fits={}
    for name, fitter in [('lbfgs',fit),('newton',fit_newton)]:
        t0=perf_counter();sc=[];ts=[];es=[];meta=[]
        before=sum(f.evaluations for fs in folds for f in fs)
        for m,fs in enumerate(folds):
            for f in fs:
                t,e,mm=fitter(f,starts[m],gtol=1e-9)
                mu,_,_=f.prediction(t,e)
                sc.append(float(np.mean((f.test[1]-mu)**2)))
                ts.append(t);es.append(e);meta.append(mm)
        times[name]=perf_counter()-t0
        evals[name]=sum(f.evaluations for fs in folds for f in fs)-before
        reference[name]=np.array(sc).reshape(len(columns),len(groups))
        fits[name]=(ts,es,meta)
    approximation={}
    for name, steps in [('frozen',0),('one_step',1)]:
        t0=perf_counter();sc=[]
        for m,fs in enumerate(folds):
            for f in fs:
                t=starts[m].copy();e=f.evaluate(t)
                for _ in range(steps):
                    t=newton_step(f,t,e);e=f.evaluate(t)
                mu,_,_=f.prediction(t,e)
                sc.append(float(np.mean((f.test[1]-mu)**2)))
        times[name]=perf_counter()-t0
        approximation[name]=np.array(sc).reshape(len(columns),len(groups))
    t0=perf_counter()
    before=sum(f.evaluations for fs in folds for f in fs)
    winner,states,lower,upper,rounds=adaptive(folds,starts,len(groups))
    times['adaptive']=perf_counter()-t0
    evals['adaptive']=sum(f.evaluations for fs in folds for f in fs)-before
    truth=reference['newton'];cv=truth.mean(axis=1);target=int(np.argmin(cv))
    audit=[]
    for idx,s in enumerate(states):
        c=s.cert;real=truth[s.model,s.group]
        within=True;dist=np.nan
        if c['valid']:
            ref=fits['newton'][0][idx]
            dist=float(np.linalg.norm(np.linalg.solve(c['S'],ref-s.theta)))
            within=dist <= c['outer_radius']+1e-7
        audit.append({'regime':regime,'seed':seed,'model':s.model,
                      'group':int(groups[s.group]),'n_test':len(s.fold.test[1]),
                      'steps':s.steps,'checks':s.checks,'local_valid':c['valid'],
                      'numerical_fallback':s.numerical,'lower':c['lo'],
                      'upper':c['hi'],'reference_loss':real,
                      'contains_reference':c['lo']-1e-8 <= real <= c['hi']+1e-8,
                      'reference_in_local_ball':within,'root_distance':dist,
                      'root_radius':c.get('outer_radius',np.nan)})
    fast_name=min(['lbfgs','newton'],key=lambda n:times[n])
    row={'regime':regime,'seed':seed,'implementation':implementation,'n':len(y),'groups':len(groups),
         'common_seconds':common_seconds,'reference_winner':target,
         'adaptive_winner':winner,'adaptive_agrees':winner==target,
         'local_certificates':sum(s.cert['valid'] for s in states),
         'numerical_fallbacks':sum(s.numerical for s in states),
         'fold_count':len(states),'refinement_rounds':rounds,
         'reference_difference':float(np.max(np.abs(reference['lbfgs']-truth))),
         'root_mismatches':sum(not x['reference_in_local_ball'] for x in audit),
         'containment_failures':sum(not x['contains_reference'] for x in audit),
         'max_reference_projected_gradient':max(x['projected_gradient'] for x in fits['newton'][2]),
         'speedup_kernel':times[fast_name]/times['adaptive'],
         'speedup_total':(common_seconds+times[fast_name])/(common_seconds+times['adaptive']),
         'fastest_reference':fast_name}
    for name,t in times.items():row[name+'_seconds']=t
    for name,n in evals.items():row[name+'_evaluations']=n
    for name,scores in approximation.items():
        approximate_cv=scores.mean(axis=1)
        row[name+'_agrees']=int(np.argmin(approximate_cv))==target
        row[name+'_max_cv_error']=float(np.max(np.abs(approximate_cv-cv)))
    for m in range(len(columns)):
        row[f'cv_{m}']=cv[m];row[f'lower_{m}']=lower[m];row[f'upper_{m}']=upper[m]
    return row,audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=10)
    parser.add_argument('--na',type=int,default=12)
    parser.add_argument('--nb',type=int,default=10)
    parser.add_argument('--regimes',nargs='+',default=['balanced','sparse','unbalanced','near_tie','boundary'])
    parser.add_argument('--out',default='../results')
    parser.add_argument('--implementation',choices=['dense','compressed'],default='compressed')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];audits=[]
    for ri,regime in enumerate(args.regimes):
        for rep in range(args.reps):
            seed=20261025+ri*1000+rep
            row,audit=trial(regime,seed,args.na,args.nb,args.implementation)
            rows.append(row);audits.extend(audit)
            pd.DataFrame(rows).to_csv(out/'runs.csv',index=False)
            pd.DataFrame(audits).to_csv(out/'fold_audit.csv',index=False)
            print(f'{regime} {rep+1}/{args.reps}: agree={row["adaptive_agrees"]} '
                  f'local={row["local_certificates"]}/{row["fold_count"]} '
                  f'speedup={row["speedup_total"]:.2f} '
                  f'failures={row["containment_failures"]}',flush=True)
    env={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
         'platform':platform.platform(),'parameters':vars(args),
         'threads':'Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1',
         'scope':'Local roots; double precision; simulated data only; '+args.implementation+' implementation',
         'timing':'Includes common fitting and fold preparation in total speedup; kernel excludes it'}
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    print(pd.DataFrame(rows).groupby('regime').agg(
        agreement=('adaptive_agrees','mean'),median_speedup=('speedup_total','median'),
        local=('local_certificates','sum'),fallback=('numerical_fallbacks','sum'),
        failures=('containment_failures','sum')).to_string(),flush=True)


if __name__=='__main__':main()
