"""Repeated comparison from a separately committed training-only plan.

Use one BLAS thread. This script never fits the full dataset for initialization.
It retains all local-test outcomes and labels fallback point scores separately.
"""
import json,sys,platform,io,contextlib
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
from acv import make_fold
from boundary_acv import adaptive,fit_box,training_start,face_certificate
from run_boundary_study import simulation
from run_lexdec import load_data,COLUMNS

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'training_timing'
METHODS=['newton','lbfgs','practical','default_lbfgs','bound_once','adaptive','norm_only']

def prepare(kind,seed):
    if kind=='large_fresh':
        X,y,a,b=simulation('balanced',seed,30,24)
        X=np.column_stack([X,np.random.default_rng(seed+9000000).normal(size=(len(y),5))])
        cols=[list(range(p)) for p in range(2,10)]
    else:
        df,X,y=load_data();first,second=('Subject','Word') if kind=='new_subject' else ('Word','Subject')
        a=df[first].to_numpy();b=df[second].to_numpy();cols=COLUMNS
    t=perf_counter()
    folds=[[make_fold(X,y,a,b,c,k,'compressed') for k in np.unique(a)] for c in cols]
    return folds,perf_counter()-t

def run(method,folds,trace=None):
    M,G=len(folds),len(folds[0])
    if method in ['adaptive','norm_only']:
        winner,states,lo,hi,rounds=adaptive(folds,starts=None,initial_budget=12,max_steps=20,
            loss_bound='directional' if method=='adaptive' else 'norm',audit_trace=trace)
        return dict(winner=winner,states=states,lower=lo,upper=hi,rounds=rounds,
          scores=np.array([s.cert['mse'] for s in states]).reshape(M,G),
          iterations=sum(s.steps+s.fallback_iterations for s in states),checks=sum(s.checks for s in states),
          valid=sum(s.cert['valid'] for s in states),fallback=sum(s.numerical for s in states))
    scores=[];certs=[];params=[];iterations=0
    for fs in folds:
        for f in fs:
            start=training_start(f)
            if method in ['newton','lbfgs']:theta,ev,meta=fit_box(f,start,method=method)
            elif method in ['practical','bound_once']:
                theta,ev,meta=fit_box(f,start,gtol=1e-6,ftol=1e-9,polish=False,terminal_hessian=method=='bound_once')
            else:
                scale=max(float(np.var(f.y)),1e-5)
                theta,ev,meta=fit_box(f,start,gtol=1e-5/scale,ftol=1e7*np.finfo(float).eps,polish=False,terminal_hessian=False)
            c=face_certificate(f,theta,ev) if method=='bound_once' else None
            scores.append(float(np.mean((f.test[1]-f.prediction(theta,ev)[0])**2)));certs.append(c);params.append(theta)
            iterations+=meta['iterations']
    scores=np.array(scores).reshape(M,G);winner=int(scores.mean(axis=1).argmin());bound_winner=None
    if method=='bound_once' and all(c['valid'] for c in certs):
        lo=np.array([c['lo'] for c in certs]).reshape(M,G).mean(axis=1)
        hi=np.array([c['hi'] for c in certs]).reshape(M,G).mean(axis=1)
        best=int(hi.argmin());other=np.delete(lo,best).min();guard=1e-10+1e-10*max(abs(hi[best]),abs(other))
        if hi[best]+guard<other:bound_winner=best
    return dict(winner=winner,scores=scores,theta=np.array(params),certs=certs,iterations=iterations,
      checks=M*G if method=='bound_once' else 0,valid=sum(c['valid'] for c in certs) if method=='bound_once' else 0,
      fallback=0,bound_winner=bound_winner,rounds=0)

def main():
    plan=json.loads((OUT/'plan.json').read_text())
    assert plan['methods']==METHODS
    jobs=[('large_fresh',s,plan['simulation_timing_repetitions']) for s in plan['simulation_seeds']]+[(t,88001+i,plan['lexdec_timing_repetitions']) for i,t in enumerate(plan['lexdec_targets'])]
    rows=[];times=[];audits=[];tests=[];decisions=[];errors=[]
    for ji,(kind,seed,repeats) in enumerate(jobs):
        tjob=perf_counter();folds,setup=prepare(kind,seed);trace={'tests':[],'decisions':[]}
        values={method:run(method,folds,trace if method=='adaptive' else None) for method in METHODS}
        ref=values['newton'];ad=values['adaptive'];best=ref['winner'];rng=np.random.default_rng(seed+700000)
        localtimes=[]
        for rep in range(repeats):
            for pos,method in enumerate(rng.permutation(METHODS)):
                t=perf_counter();value=run(method,folds);elapsed=perf_counter()-t
                rec=dict(kind=kind,seed=seed,repeat=rep,position=pos,method=method,seconds=elapsed,
                  winner=value['winner'],repeat_agrees=value['winner']==values[method]['winner'])
                times.append(rec);localtimes.append(rec)
        med=pd.DataFrame(localtimes).groupby('method').seconds.median();n=len(folds)*len(folds[0])
        row=dict(kind=kind,seed=seed,folds=n,models=len(folds),groups=len(folds[0]),common_seconds=setup,
          reference_winner=best,adaptive_agrees=ad['winner']==best,adaptive_valid=ad['valid'],fallbacks=ad['fallback'],
          adaptive_rounds=ad['rounds'],norm_rounds=values['norm_only']['rounds'],
          bound_once_resolves=values['bound_once']['bound_winner']==best,
          bound_once_valid=values['bound_once']['valid'],anchor_fallbacks=sum(s.fallback_reason=='root_anchor' for s in ad['states']))
        for method,val in values.items():
            row[method+'_seconds']=float(med[method]);row[method+'_agrees']=val['winner']==best
            row[method+'_iterations']=val['iterations'];row[method+'_checks']=val['checks']
            row[method+'_max_cv_error']=float(np.max(abs(val['scores'].mean(axis=1)-ref['scores'].mean(axis=1))))
        strict=min(med['newton'],med['lbfgs'])
        row['strict_kernel_ratio']=strict/med['adaptive'];row['strict_total_ratio']=(setup+strict)/(setup+med['adaptive'])
        for method in ['practical','default_lbfgs','bound_once','norm_only']:
            row[method+'_kernel_ratio']=med[method]/med['adaptive'];row[method+'_total_ratio']=(setup+med[method])/(setup+med['adaptive'])
        rows.append(row)
        for i,s in enumerate(ad['states']):
            c=s.cert;actual=ref['scores'][s.model,s.group];theta=ref['theta'][i]
            a=dict(kind=kind,seed=seed,model=s.model,group=s.group,valid=c['valid'],fallback=s.numerical,
              fallback_reason=s.fallback_reason,lower=c['lo'],upper=c['hi'],reference_loss=actual,
              approximate_loss=c['mse'],steps=s.steps,checks=s.checks,excess=max(c['lo']-actual,actual-c['hi'],0.))
            if c['valid']:
                free=c['free'];a['active_match']=bool(np.array_equal(s.theta==0,theta==0))
                a['root_in_r']=bool(np.linalg.norm(np.linalg.solve(c['S'][free],(theta-s.theta)[free]))<=c['outer_radius'] and a['active_match'])
            for method,v in values.items():a[method+'_loss']=v['scores'][s.model,s.group]
            audits.append(a)
        tests.extend(dict(kind=kind,seed=seed,**t) for t in trace['tests']);decisions.extend(dict(kind=kind,seed=seed,**d) for d in trace['decisions'])
        for name,data in [('runs',rows),('timings',times),('folds',audits),('tests',tests),('decisions',decisions)]:
            pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
        print(f'{ji+1}/{len(jobs)} {kind} {seed}: agrees={row["adaptive_agrees"]}, fallback={ad["fallback"]}, total practical/adaptive={row["practical_total_ratio"]:.3f}, bound/adaptive={row["bound_once_total_ratio"]:.3f}, job={perf_counter()-tjob:.1f}s',flush=True)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):np.show_config()
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,platform=platform.platform(),blas_threads=1,numpy_config=buf.getvalue()),indent=2))

if __name__=='__main__':main()
