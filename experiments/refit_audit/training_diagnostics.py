"""Training-only diagnostics for first checks in the large simulation study."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cases import HERE, ROOT
from acv import make_fold
from boundary_acv import face_certificate, fit_box, training_start
from run_boundary_study import simulation


def one(fold):
    center,ev,_=fit_box(fold,training_start(fold),gtol=1e-6,ftol=1e-9,polish=False)
    cert=face_certificate(fold,center,ev)
    ref,ref_ev,_=fit_box(fold,training_start(fold),method='newton')
    if not cert['valid']:
        return dict(accepted=False,reason=cert['reason'])
    free=cert['free']
    delta=np.linalg.solve(cert['S'][free],(ref-center)[free])
    distance=np.linalg.norm(delta)
    rho=cert['radius']
    mu,geo=fold.prediction_geometry(center,ev,cert['S'])
    residual=fold.test[1]-mu
    gamma=residual @ (geo @ ev.w).T
    norm_grad=np.linalg.norm(gamma)
    actual=float(np.mean((fold.test[1]-fold.prediction(ref,ref_ev)[0])**2))
    err=abs(cert['mse']-actual)
    denom=norm_grad*distance
    remainder=max(0.,cert['Gamma']-rho*norm_grad)
    h=ev.hessian[np.ix_(free,free)]
    return dict(accepted=True,same_face=np.array_equal(np.flatnonzero(ref==0),cert['active']),
        in_ball=bool(distance<=cert['outer_radius']),half_width=(cert['hi']-cert['lo'])/2,
        loss=cert['mse'],reference_loss=actual,loss_error=err,usable=err>1e-12*max(1.,actual),
        hessian_condition=np.linalg.cond(h),smallest_free_over_residual=np.min(center[free])/center[0],
        active_count=len(cert['active']),a=cert['a'],Lr=cert['Lr'],c=cert['c'],c0=cert['c0'],
        rho=rho,eta=cert['eta'],root_distance=distance,
        radius_over_distance=rho/distance if distance else np.nan,
        alignment=abs(gamma@delta)/denom if denom else np.nan,
        gamma_remainder_fraction=remainder/cert['Gamma'] if cert['Gamma'] else np.nan)


def main(output):
    output.mkdir(exist_ok=True,parents=True)
    seeds=pd.read_csv(ROOT/'results/training_timing/runs.csv').query("kind == 'large_fresh'").seed.unique()
    rows=[]
    for seed in sorted(seeds):
        X,y,a,b=simulation('balanced',int(seed),30,24)
        X=np.column_stack([X,np.random.default_rng(int(seed)+9000000).normal(size=(len(y),5))])
        for m,p in enumerate(range(2,10)):
            for k,g in enumerate(np.unique(a)):
                fold=make_fold(X,y,a,b,list(range(p)),g,'compressed')
                rows.append(dict(seed=seed,model=m,fold=k,**one(fold)))
        print('finished',seed,flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(output/'training_geometry.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    g=raw[raw.accepted & raw.same_face & raw.in_ball & raw.usable].copy()
    g['ratio']=g.half_width/g.loss_error
    g['inverse_alignment']=1/g.alignment
    measures=['hessian_condition','smallest_free_over_residual','a','Lr','c','alignment','radius_over_distance','gamma_remainder_fraction']
    corr={v:float(spearmanr(g.ratio,g[v],nan_policy='omit').statistic) for v in measures}
    cols=['ratio']+measures
    summary=dict(folds=len(raw),valid=int(raw.accepted.sum()),usable=len(g),active=int(raw.active_count.gt(0).sum()),
        quantiles={v:{str(q):float(g[v].quantile(q)) for q in [.5,.95,1.]} for v in cols},spearman=corr)
    (output/'training_geometry.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    main(parser.parse_args().output)
