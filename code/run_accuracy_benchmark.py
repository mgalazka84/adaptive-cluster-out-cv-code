"""Accuracy, work and runtime benchmarks using the released default tie guard.

The fixed plan is recorded before outcomes in results/accuracy_v2/<suite>/plan.json.
Fold-difference SEs are descriptive iid-fold proxies; training sets overlap.
No statistical confidence claim is based on those proxies.
"""
import argparse, json, platform, sys
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
from acv import CompressedFold, make_fold
from boundary_acv import adaptive, fit_box, active_step, face_certificate, boundary_tube, reporting_radius
from run_boundary_study import simulation
from run_lexdec import make_problem

ROOT=Path(__file__).resolve().parents[1]
METHODS=['newton','lbfgs','practical','default_lbfgs','loose','certify_once',
         'adaptive','norm_only','interior_only','frozen','one_step']

def run(name, folds, starts, audit_trace=None):
    M,G=len(folds),len(folds[0])
    if name in ['adaptive','norm_only','interior_only']:
        winner,states,lo,hi,rounds=adaptive(folds,starts,allow_boundary=name!='interior_only',
            loss_bound='norm' if name=='norm_only' else 'directional',audit_trace=audit_trace)
        return dict(winner=winner,states=states,lower=lo,upper=hi,rounds=rounds,
                    scores=np.array([s.cert['mse'] for s in states]).reshape(M,G),
                    iterations=[s.steps+s.fallback_iterations for s in states],
                    checks=[s.checks for s in states],
                    fallback=sum(s.numerical for s in states))
    scores=[];params=[];metas=[];certs=[]
    for m,fs in enumerate(folds):
        for fold in fs:
            if name in ['newton','lbfgs']:
                theta,ev,meta=fit_box(fold,starts[m],method=name)
            elif name in ['practical','certify_once']:
                theta,ev,meta=fit_box(fold,starts[m],gtol=1e-6,ftol=1e-9,polish=False,
                                    terminal_hessian=name=='certify_once')
            elif name=='default_lbfgs':
                scale=max(float(np.var(fold.y)),1e-5)
                theta,ev,meta=fit_box(fold,starts[m],gtol=1e-5/scale,
                                    ftol=1e7*np.finfo(float).eps,polish=False,terminal_hessian=False)
            elif name=='loose':
                theta,ev,meta=fit_box(fold,starts[m],gtol=1e-3,ftol=1e-5,polish=False,terminal_hessian=False)
            else:
                theta=starts[m].copy();ev=fold.evaluate(theta)
                if name=='one_step':theta=active_step(fold,theta,ev);ev=fold.evaluate(theta)
                meta={'iterations':int(name=='one_step')}
            c=face_certificate(fold,theta,ev) if name=='certify_once' else None
            loss=float(np.mean((fold.test[1]-fold.prediction(theta,ev)[0])**2))
            scores.append(loss);params.append(theta);metas.append(meta);certs.append(c)
    scores=np.array(scores).reshape(M,G)
    winner=int(np.argmin(scores.mean(axis=1)))
    result=dict(scores=scores,theta=np.array(params),metas=metas,certs=certs,winner=winner,
                iterations=[m['iterations'] for m in metas],checks=[int(name=='certify_once')]*len(metas),
                fallback=0)
    if name=='certify_once':
        valid=all(c['valid'] for c in certs)
        certified_winner=None
        if valid:
            low=np.array([c['lo'] for c in certs]).reshape(M,G).mean(axis=1)
            high=np.array([c['hi'] for c in certs]).reshape(M,G).mean(axis=1)
            best=int(np.argmin(high));other=np.min(np.delete(low,best))
            if high[best]+1e-10+1e-10*max(abs(high[best]),abs(other))<other:certified_winner=best
        result.update(certified_winner=certified_winner,valid_fits=sum(c['valid'] for c in certs))
    return result

def problem(regime,seed,na=12,nb=10,models=3):
    X,y,a,b=simulation(regime if regime!='many_levels' else 'balanced',seed,na,nb)
    if models>3:
        rng=np.random.default_rng(seed+9000000)
        X=np.column_stack([X,rng.normal(size=(len(y),models-3))])
    cols=[list(range(p)) for p in range(2,models+2)]
    t=perf_counter()
    starts=[fit_box(CompressedFold(X[:,c],y,a,b),[1.,.7,.5])[0] for c in cols]
    folds=[[make_fold(X,y,a,b,c,k,'compressed') for k in np.unique(a)] for c in cols]
    return folds,starts,perf_counter()-t

def benchmark(folds,starts,setup,seed,repeats,audit=True):
    trace={'tests':[],'decisions':[]}
    canonical={name:run(name,folds,starts,trace if name=='adaptive' else None) for name in METHODS}
    rng=np.random.default_rng(seed+700000)
    timing=[]
    for rep in range(repeats):
        for pos,name in enumerate(rng.permutation(METHODS)):
            t=perf_counter();value=run(name,folds,starts);elapsed=perf_counter()-t
            timing.append(dict(repeat=rep,position=pos,method=name,seconds=elapsed))
            assert value['winner']==canonical[name]['winner']
    med=pd.DataFrame(timing).groupby('method').seconds.median()
    ref=canonical['newton'];ad=canonical['adaptive'];score=ref['scores'];cv=score.mean(axis=1)
    order=np.argsort(cv);best,second=map(int,order[:2]);diff=score[second]-score[best]
    fold_se=float(np.std(diff,ddof=1)/np.sqrt(len(diff)))
    summary=dict(reference_winner=best,adaptive_winner=ad['winner'],adaptive_agrees=ad['winner']==best,
                 cv_gap=float(cv[second]-cv[best]),paired_fold_se_proxy=fold_se,
                 gap_to_se_proxy=float((cv[second]-cv[best])/fold_se),
                 common_seconds=setup,fold_count=score.size,
                 local_certificates=sum(s.cert['valid'] for s in ad['states']),
                 active_face_certificates=sum(s.cert.get('reason')=='active_face' for s in ad['states']),
                 numerical_fallbacks=ad['fallback'],interior_only_fallbacks=canonical['interior_only']['fallback'],
                 refinement_rounds=ad['rounds'],
                 norm_refinement_rounds=canonical['norm_only']['rounds'],
                 tie_guard_binding_events=sum(d['zero_resolves'] and not d['default_resolves'] for d in trace['decisions']),
                 cert_once_valid_fits=canonical['certify_once']['valid_fits'],
                 cert_once_resolves=canonical['certify_once']['certified_winner']==best,
                 independent_ref_max_loss_difference=float(np.max(np.abs(canonical['lbfgs']['scores']-score))))
    for name,result in canonical.items():
        summary[name+'_seconds']=float(med[name])
        summary[name+'_iterations']=float(np.mean(result['iterations']))
        summary[name+'_checks']=float(np.mean(result['checks']))
        summary[name+'_agrees']=result['winner']==best
        summary[name+'_max_cv_error']=float(np.max(np.abs(result['scores'].mean(axis=1)-cv)))
    for m in range(len(folds)):
        summary[f'cv_{m}']=float(cv[m]);summary[f'lower_{m}']=float(ad['lower'][m]);summary[f'upper_{m}']=float(ad['upper'][m])
    summary['strict_speedup_total']=(setup+min(med['newton'],med['lbfgs']))/(setup+med['adaptive'])
    summary['practical_speedup_total']=(setup+med['practical'])/(setup+med['adaptive'])
    summary['default_speedup_total']=(setup+med['default_lbfgs'])/(setup+med['adaptive'])
    summary['interior_only_speedup_total']=(setup+min(med['newton'],med['lbfgs']))/(setup+med['interior_only'])
    rows=[]
    for i,s in enumerate(ad['states']):
        c=s.cert;m,k=s.model,s.group;theta_ref=ref['theta'][i];loss_ref=score[m,k]
        row=dict(model=m,group_index=k,local_valid=c['valid'],active_count=len(c.get('active',[])),
                 lower=c['lo'],upper=c['hi'],reference_loss=loss_ref,approximate_loss=c['mse'],
                 numerical_fallback=s.numerical,steps=s.steps,checks=s.checks,
                 reference_gradient=ref['metas'][i]['projected_gradient'])
        for name in METHODS:
            row[name+'_loss']=float(canonical[name]['scores'][m,k])
        row['contained']=c['lo']-1e-8<=loss_ref<=c['hi']+1e-8
        excess=max(c['lo']-loss_ref,loss_ref-c['hi'],0.)
        row.update(contained_no_tolerance=excess==0,containment_excess=excess,
                   containment_excess_relative_width=excess/(c['hi']-c['lo']) if c['hi']>c['lo'] else (np.inf if excess else 0.))
        for j in range(3):row[f'theta_{j}']=s.theta[j];row[f'reference_theta_{j}']=theta_ref[j]
        if c['valid']:
            free=c['free'];act=c['active']
            err=float(np.linalg.norm(np.linalg.solve(c['S'][free],(theta_ref-s.theta)[free])))
            le=abs(loss_ref-c['mse']);width=(c['raw_hi']-c['raw_lo'])/2
            row.update(parameter_error=err,rho=c['radius'],parameter_ratio=c['radius']/err if err>1e-10 else np.nan,
                       loss_error=le,loss_half_width=width,loss_ratio=width/le if le>1e-12*max(1,abs(loss_ref)) else np.nan,
                       active_match=np.array_equal(np.flatnonzero(theta_ref==0),act),
                       reference_in_ball=bool(err<=c['outer_radius']+1e-7 and np.all(theta_ref[act]==0)),
                       reference_in_rho=bool(err<=c['radius'] and np.all(theta_ref[act]==0)),
                       reference_in_r=bool(err<=c['outer_radius'] and np.all(theta_ref[act]==0)),
                       outer_radius=c['outer_radius'],eta=c['eta'],self_map=c['self_map'],
                       norm_loss_half_width=(c['norm_hi']-c['norm_lo'])/2,
                       Lr=c['Lr'],a=c['a'],f2=c['f2'],b0=c['b0'],b_r=c['b_r'],c=c['c'],c0=c['c0'],
                       trace_L=c['trace_L'],quadratic_L=c['quadratic_L'],
                       active_margin_fraction=min((lower/s.ev.gradient[j] for j,lower in zip(act,c['active_lower'])),default=np.nan))
            row['reporting_radius']=reporting_radius(s.ev,c)
            if len(act):
                tube=boundary_tube(s.ev,c)
                larger=boundary_tube(s.ev,dict(c,outer_radius=row['reporting_radius']))
                row.update(tube_epsilon=tube['active_epsilon'] if tube else np.nan,
                           tube_free_radius=tube['free_radius'] if tube else np.nan,
                           larger_tube_epsilon=larger['active_epsilon'] if larger else np.nan,
                           larger_tube_free_radius=larger['free_radius'] if larger else np.nan)
        if audit:
            scale=max(float(np.var(s.fold.y)),1e-4)
            audit_fits=[fit_box(s.fold,scale*np.array(v)) for v in ([.5,.25,.25],[.8,.1,.1],[.1,.45,.45])]
            row['training_start_active_mismatches']=sum(not np.array_equal(t==0,theta_ref==0) for t,e,meta in audit_fits)
            t,e,meta=min(audit_fits,key=lambda x:x[1].value)
            row['training_start_loss']=float(np.mean((s.fold.test[1]-s.fold.prediction(t,e)[0])**2))
        rows.append(row)
    summary['containment_failures']=sum(not r['contained'] for r in rows)
    summary['root_mismatches']=sum(not r.get('reference_in_ball',True) for r in rows)
    if audit:
        losses=np.array([r['training_start_loss'] for r in rows]).reshape(score.shape)
        summary['training_start_agrees']=int(np.argmin(losses.mean(axis=1)))==best
        summary['training_start_active_mismatches']=sum(r['training_start_active_mismatches'] for r in rows)
    # Microbenchmarks at final adaptive centers; include evaluation in a Newton step.
    micro=[]
    for index in np.unique(np.linspace(0,len(ad['states'])-1,min(5,len(ad['states'])),dtype=int)):
        s=ad['states'][index];f=s.fold;t=s.theta
        a=[];b=[]
        for _ in range(5):
            ev=f.evaluate(t)
            t0=perf_counter();next_theta=active_step(f,t,ev);f.evaluate(next_theta);a.append(perf_counter()-t0)
            t0=perf_counter();face_certificate(f,t,ev);b.append(perf_counter()-t0)
        micro.append(dict(model=s.model,group_index=s.group,newton_seconds=float(np.median(a)),certificate_seconds=float(np.median(b))))
    return summary,rows,timing,micro,trace

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['original','scale','fresh','lexdec'],required=True)
    p.add_argument('--timing-reps',type=int,default=3);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--output-root',default='accuracy_v2')
    args=p.parse_args();out=ROOT/'results'/args.output_root/args.suite;out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    regimes=['balanced','sparse','unbalanced','near_tie','boundary','zero_a','zero_both']
    if args.suite=='original':jobs=[(r,20261025+1000*ri+i,12,10,3) for ri,r in enumerate(regimes) for i in range(20)]
    elif args.suite=='scale':jobs=[(r,20261025+1000*ri+i,24,20,3) for ri,r in enumerate(['balanced','boundary','zero_both']) for i in range(3)]
    elif args.suite=='fresh':jobs=[(r,74000000+1000*ri+i,*( (30,24,8) if r=='many_levels' else (12,10,3))) for ri,r in enumerate(['near_tie','boundary','zero_both','many_levels']) for i in range(20)]
    else:jobs=[(r,88001+i,0,0,3) for i,r in enumerate(['new_subject','new_word'])]
    if args.limit:jobs=jobs[:args.limit]
    plan={'suite':args.suite,'jobs':jobs,'timing_repeats':args.timing_reps,'methods':METHODS,
          'selection_atol':1e-10,'selection_rtol':1e-10,'interval_intersection':True,
          'loss_bound':'directional','unpolished_terminal_hessian':False,
          'fresh_seeds_fixed_before_results':True,'fold_se_scope':'descriptive iid-fold proxy, not a calibrated CV standard error',
          'practical':{'gtol_original':1e-6,'ftol':1e-9,'polish':False},
          'default_lbfgs':{'gtol_scaled':1e-5,'ftol':1e7*np.finfo(float).eps,'polish':False},
          'loose':{'gtol_original':1e-3,'ftol':1e-5,'polish':False}}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    records=[];audits=[];times=[];micros=[];tests=[];decisions=[]
    for index,(regime,seed,na,nb,models) in enumerate(jobs):
        if args.suite=='lexdec':folds,starts,setup,_=make_problem(regime)
        else:folds,starts,setup=problem(regime,seed,na,nb,models)
        summary,ar,tr,mr,trace=benchmark(folds,starts,setup,seed,args.timing_reps)
        ident=dict(suite=args.suite,regime=regime,seed=seed)
        summary.update(**ident,n=sum(len(f.test[1]) for f in folds[0]),models=len(folds),groups=len(folds[0]))
        records.append(summary);audits.extend(dict(**ident,**r) for r in ar)
        times.extend(dict(**ident,**r) for r in tr);micros.extend(dict(**ident,**r) for r in mr)
        tests.extend(dict(**ident,**r) for r in trace['tests']);decisions.extend(dict(**ident,**r) for r in trace['decisions'])
        for filename,data in [('runs.csv',records),('fold_audit.csv',audits),('timings.csv',times),('microbenchmarks.csv',micros),('certificate_tests.csv',tests),('scheduler_decisions.csv',decisions)]:
            pd.DataFrame(data).to_csv(out/filename,index=False)
        print(f'{args.suite} {index+1}/{len(jobs)} {regime} agree={summary["adaptive_agrees"]} fallback={summary["numerical_fallbacks"]} speed={summary["strict_speedup_total"]:.2f}/{summary["practical_speedup_total"]:.2f}',flush=True)
    cpu='unknown'
    try:
        cpu=next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
    except (OSError,StopIteration):pass
    (out/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        pandas=pd.__version__,platform=platform.platform(),cpu=cpu,blas_threads=1),indent=2))

if __name__=='__main__':main()
