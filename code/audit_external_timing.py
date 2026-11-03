"""Prespecified modest statsmodels timing audit and real-data refinement trace.

Three models and first/middle/last sorted groups for both lexdec targets.
Fit plus prediction timings include method-specific preparation, exclude CSV
loading and common row slicing. This subset does not estimate whole-CV time.
"""
import json,warnings
from time import perf_counter
import numpy as np
import pandas as pd
import statsmodels
from acv import CompressedFold
from boundary_acv import fit_box,training_start,face_certificate,State
from audit_statsmodels import external_fit
from run_lexdec import load_data,COLUMNS
from run_revision_study import OUT

def main():
    df,X,y=load_data();rows=[];trace=[];rng=np.random.default_rng(93007000)
    for target,first,second in [('new_subject','Subject','Word'),('new_word','Word','Subject')]:
        a=df[first].to_numpy();b=df[second].to_numpy();groups=np.unique(a)
        for k in [0,len(groups)//2,len(groups)-1]:
            tr=a!=groups[k];te=~tr
            for m,cols in enumerate(COLUMNS):
                args=(X[tr][:,cols],y[tr],a[tr],b[tr]);test=(X[te][:,cols],y[te],a[te],b[te])
                for rep in range(3):
                    for pos,method in enumerate(rng.permutation(['practical','bound_once','statsmodels'])):
                        t0=perf_counter();f=CompressedFold(*args,test=test)
                        # The external model operates on observation-space arrays;
                        # discard the prototype's preparation from its timed block.
                        if method=='statsmodels':
                            t0=perf_counter();theta,mu,res,warn=external_fit(f)
                            converged=bool(res.converged);valid=None
                        else:
                            theta,ev,meta=fit_box(f,training_start(f),gtol=1e-6,ftol=1e-9,polish=False,terminal_hessian=method=='bound_once')
                            mu=f.prediction(theta,ev)[0];warn=[];converged=meta['optimizer_success']
                            valid=face_certificate(f,theta,ev)['valid'] if method=='bound_once' else None
                        elapsed=perf_counter()-t0
                        rows.append(dict(target=target,group_index=k,model=m,repeat=rep,position=pos,method=method,
                          seconds=elapsed,loss=float(np.mean((test[1]-mu)**2)),converged=converged,local_valid=valid,warnings=' | '.join(warn)))
                if m==2 and k==0:
                    f=CompressedFold(*args,test=test);s=State(f,training_start(f),m,k,initial_budget=12,max_steps=20)
                    # Deliberate diagnostic refinement after acceptance, up to 10
                    # steps; this trajectory is not charged to scheduler timings.
                    for _ in range(10):
                        s.refine();c=s.cert
                        trace.append(dict(target=target,model=m,group_index=k,step=s.steps,valid=c['valid'],reason=c['reason'],
                          lower=c.get('lo',np.nan),upper=c.get('hi',np.nan),mse=c.get('mse',np.nan)))
                        if s.numerical:break
                pd.DataFrame(rows).to_csv(OUT/'external_timings.csv',index=False)
                pd.DataFrame(trace).to_csv(OUT/'lexdec_trace.csv',index=False)
                print(target,k,m,'complete',flush=True)
    (OUT/'external_environment.json').write_text(json.dumps({'statsmodels':statsmodels.__version__,
       'optimizer':'BFGS gtol=1e-10 maxiter=500, default training-only initialization',
       'scope':'Strict independent software audit, not a claim of speedup over package defaults.'},indent=2))

if __name__=='__main__':main()
