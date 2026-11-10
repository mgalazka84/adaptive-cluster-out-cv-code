"""Four training-only starts on every lexdec and OrchardSprays fold."""
import argparse
import json
import numpy as np
import pandas as pd

from cases import HERE, CompressedFold, cases, strict_starts, branch_match
from boundary_acv import face_certificate, fit_box, projected_gradient, training_start


def orchard_design_singular_values(inputs):
    x, _, a, b, _ = inputs
    n,p=x.shape
    k=np.linalg.qr(x,mode='complete')[0][:,p:]
    parts=[np.eye(n),np.equal.outer(a,a),np.equal.outer(b,b)]
    columns=np.stack([(k.T @ v @ k).ravel() for v in parts],axis=1)
    columns/=np.linalg.norm(columns,axis=0)
    return np.linalg.svd(columns,compute_uv=False)


def main(output):
    output.mkdir(exist_ok=True,parents=True)
    rows=[]
    for name in ['lexdec','orchard']:
        for key,inputs,_ in cases(name):
            fold=CompressedFold(*inputs)
            center,ev,_=fit_box(fold,training_start(fold),gtol=1e-6,ftol=1e-9,polish=False)
            cert=face_certificate(fold,center,ev)
            if not cert['valid']:
                raise RuntimeError(('The original practical test is inconclusive',key))
            ident=orchard_design_singular_values(inputs) if name=='orchard' else None
            for start,theta0 in strict_starts(fold).items():
                ref,fit,_=fit_box(fold,theta0,method='newton')
                pred=fold.prediction(ref,fit)[0]
                rows.append(dict(**key,start=start,residual=ref[0],var_a=ref[1],var_b=ref[2],
                    objective=fit.value,mse=float(np.mean((fold.test[1]-pred)**2)),
                    projected_gradient=float(np.linalg.norm(projected_gradient(ref,fit.gradient),np.inf)),
                    same_face_and_ball=branch_match(center,ref,cert),
                    design_rank=len(ident[ident>1e-10]) if ident is not None else np.nan,
                    design_min_singular=min(ident) if ident is not None else np.nan))
    df=pd.DataFrame(rows)
    df.to_csv(output/'restarts.csv',index=False)
    summary={}
    for name,g in df.groupby('dataset'):
        gb=g.groupby(['target','model','fold'])
        summary[name]=dict(fits=len(gb),starts=len(g),same_face_and_ball=int(g.same_face_and_ball.sum()),
            max_fold_objective_range=float((gb.objective.max()-gb.objective.min()).max()),
            max_fold_loss_range=float((gb.mse.max()-gb.mse.min()).max()),
            max_projected_gradient=float(g.projected_gradient.max()))
        if name=='orchard':
            summary[name].update(min_design_rank=int(g.design_rank.min()),
                min_normalised_design_singular=float(g.design_min_singular.min()))
    (output/'restarts_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(summary)

if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    main(parser.parse_args().output)
