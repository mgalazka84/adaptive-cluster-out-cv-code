"""Fixed supplemental experiments; no tuning after inspecting their outcomes.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Archived three-component
benchmarks remain unchanged. Default training-only starts use budgets 12/20;
the historical full-data warm-start budgets were 4/7. Numerical fallback
scores never count as successful local-bound tests.
"""
import json,sys,platform,hashlib,warnings
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
from scipy.optimize._numdiff import approx_derivative
from acv import CompressedFold
from linear_psd_fold import LinearPSDFold
from boundary_acv import adaptive,fit_box,training_start,active_gradient_lower,face_certificate
from run_accuracy_benchmark import problem

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'revision_study'

def slope_problem(seed):
    rng=np.random.default_rng(seed);na,nb,rep=8,6,2
    a=np.repeat(np.arange(na),nb*rep);b=np.tile(np.repeat(np.arange(nb),rep),na)
    x=rng.normal(size=len(a));z=rng.normal(size=len(a))
    Z0=np.equal.outer(a,np.arange(na)).astype(float);Z1=np.equal.outer(b,np.arange(nb)).astype(float)
    Z=[Z0,Z1,x[:,None]*Z0,x[:,None]*Z1]
    v=[.4,.3,.15,.1] if seed%3==0 else ([.4,.3,0.,.1] if seed%3==1 else [.1,0.,0.,0.])
    y=1+.6*x+.15*z+rng.normal(size=len(a))
    for zz,vv in zip(Z,v):y+=zz@(rng.normal(size=zz.shape[1])*np.sqrt(vv))
    X=np.column_stack([np.ones(len(a)),x,z]);folds=[];starts=[]
    for p in [2,3]:
        full=LinearPSDFold(X[:,:p],y,Z,(X[:1,:p],y[:1],[zz[:1] for zz in Z]))
        starts.append(fit_box(full,training_start(full),method='newton')[0])
        fs=[]
        for k in range(na):
            tr=a!=k;te=~tr
            fs.append(LinearPSDFold(X[tr,:p],y[tr],[zz[tr] for zz in Z],(X[te,:p],y[te],[zz[te] for zz in Z])))
        folds.append(fs)
    return folds,starts

def references(folds,starts):
    scores=[];params=[];full_scores=[];full_params=[]
    for m,fs in enumerate(folds):
        for f in fs:
            t,e,_=fit_box(f,training_start(f),method='newton');params.append(t)
            scores.append(np.mean((f.test[1]-f.prediction(t,e)[0])**2))
            t,e,_=fit_box(f,starts[m],method='newton');full_params.append(t)
            full_scores.append(np.mean((f.test[1]-f.prediction(t,e)[0])**2))
    return np.array(scores).reshape(len(folds),-1),np.array(params),np.array(full_scores).reshape(len(folds),-1),np.array(full_params)

def summarize(folds,starts,ref,pars,seed,kind,setting,opts):
    trace={'tests':[],'decisions':[]};t=perf_counter()
    winner,states,lo,hi,rounds=adaptive(folds,**opts,audit_trace=trace)
    seconds=perf_counter()-t;best=int(ref.mean(axis=1).argmin());rows=[]
    for i,s in enumerate(states):
        c=s.cert;actual=ref[s.model,s.group]
        row=dict(seed=seed,kind=kind,setting=setting,model=s.model,group=s.group,
          valid=c['valid'],fallback=s.numerical,fallback_reason=s.fallback_reason,
          steps=s.steps,checks=s.checks,lower=c['lo'],upper=c['hi'],reference_loss=actual,
          approximate_loss=c['mse'],excess=max(c['lo']-actual,actual-c['hi'],0.),
          active_count=len(c.get('active',[])))
        if c['valid']:
            pairs=[active_gradient_lower(s.ev,j,c['a']*c['radius'],True) for j in c['active']]
            le=abs(actual-c['mse']);half=(c['hi']-c['lo'])/2
            row.update(second_branch_wins=sum(b>a for a,b in pairs),branch_checks=len(pairs),
              radius=c['radius'],Lr=c['Lr'],loss_ratio=half/le if le>1e-12*max(1,abs(actual)) else np.nan,
              active_match=np.array_equal(s.theta==0,pars[i]==0))
        rows.append(row)
    summary=dict(seed=seed,kind=kind,setting=setting,components=len(folds[0][0].C),folds=len(states),
       winner=winner,reference_winner=best,agrees=winner==best,seconds=seconds,rounds=rounds,
       valid=sum(s.cert['valid'] for s in states),fallbacks=sum(s.numerical for s in states),
       checks=sum(s.checks for s in states),steps=sum(s.steps for s in states),
       max_excess=max(r['excess'] for r in rows),
       active_not_separated=sum(t['reason']=='active_gradient_not_separated' for t in trace['tests']))
    return summary,rows,trace

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    settings={
      'default':{},'budget_4_7':{'initial_budget':4,'max_steps':7},
      'budget_8_14':{'initial_budget':8,'max_steps':14},
      'budget_16_28':{'initial_budget':16,'max_steps':28},
      'guard_zero':{'selection_atol':0.,'selection_rtol':0.},
      'guard_1e-8':{'selection_atol':1e-8,'selection_rtol':1e-8},
      'floor_low':{'step_options':{'eigen_abs':.001,'eigen_rel':1e-6},'bound_options':{'curvature_floor':1e-12}},
      'floor_high':{'step_options':{'eigen_abs':.1,'eigen_rel':1e-4},'bound_options':{'curvature_floor':1e-8}},
      'radius_1.5':{'bound_options':{'radius_factor':1.5}},
      'radius_3':{'bound_options':{'radius_factor':3.}},
      'armijo_1e-3':{'step_options':{'armijo':1e-3}}}
    jobs=[('intercept',s) for s in range(93003000,93003008)]+[('slope',s) for s in range(93005000,93005012)]
    plan={'jobs':jobs,'settings':settings,'training_only_default_budgets':[12,20],
      'sensitivity_seeds':list(range(93003000,93003008)),
      'timings':'Single-run supplemental diagnostics, not speedup evidence',
      'external_plan':{'targets':['new_subject','new_word'],'groups':'first, middle, last (sorted)','models':[0,1,2],'repeats':3},
      'version':'Fixed before first execution; no discarded seeds.'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    env={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
      'platform':platform.platform(),'blas_threads':1}
    import io,contextlib
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):np.show_config()
    env['numpy_config']=buf.getvalue();(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    runs=[];audits=[];branches=[];traces=[];derivative_checks=[]
    for kind,seed in jobs:
        if kind=='slope':folds,starts=slope_problem(seed)
        else:folds,starts,_=problem(['balanced','boundary','zero_both','near_tie'][seed%4],seed)
        ref,pars,full,fullpars=references(folds,starts)
        branches.append(dict(seed=seed,kind=kind,training_winner=int(ref.mean(axis=1).argmin()),full_winner=int(full.mean(axis=1).argmin()),
          active_mismatches=int(np.sum(np.any((pars==0)!=(fullpars==0),axis=1))),max_loss_difference=float(np.max(np.abs(ref-full)))))
        if kind=='slope' and seed==jobs[8][1]:
            f=folds[0][0];t=np.array([1.,.4,.3,.15,.1]);ev=f.evaluate(t)
            ng=approx_derivative(lambda th:np.array([f.objective(th)]),t,method='3-point').ravel()
            nh=approx_derivative(lambda th:f.evaluate(th).gradient,t,method='3-point')
            derivative_checks.append(dict(gradient_max_abs=float(abs(ng-ev.gradient).max()),hessian_max_abs=float(abs(nh-ev.hessian).max())))
            assert np.allclose(ng,ev.gradient,rtol=2e-6,atol=2e-6)
            assert np.allclose(nh,ev.hessian,rtol=2e-6,atol=2e-6)
        chosen=settings if kind=='intercept' else {'default':{}}
        for setting,opts in chosen.items():
            summary,rows,trace=summarize(folds,starts,ref,pars,seed,kind,setting,opts)
            runs.append(summary);audits.extend(rows)
            if setting=='default':
                traces.extend(dict(seed=seed,kind=kind,**t) for t in trace['tests'])
        for name,data in [('runs',runs),('folds',audits),('starts',branches),('trace',traces)]:pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
        print(kind,seed,'valid',runs[-1]['valid'],'fallbacks',runs[-1]['fallbacks'],'agrees',runs[-1]['agrees'],flush=True)
    (OUT/'derivative_checks.json').write_text(json.dumps(derivative_checks,indent=2))

if __name__=='__main__':main()
