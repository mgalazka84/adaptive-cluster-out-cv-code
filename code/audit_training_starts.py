"""Audit warm-start references against three TRAINING-ONLY starts per fold.

This empirical audit is not a proof that the REML optimum is global.
The restart timing is deliberately excluded from claimed adaptive speedups.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from acv import simulate,make_fold,fit


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='research_acv/results/compressed')
    args=p.parse_args();root=Path(args.results)
    runs=pd.read_csv(root/'runs.csv');old=pd.read_csv(root/'fold_audit.csv')
    rows=[]
    for _,run in runs.iterrows():
        X,y,a,b=simulate(int(run.seed),run.regime,na=12,nb=10)
        groups=np.unique(a)
        for m,cols in enumerate([[0,1],[0,1,2],[0,1,2,3]]):
            for group in groups:
                f=make_fold(X,y,a,b,cols,group,'compressed')
                scale=max(.1,float(np.var(f.y)))
                starts=[scale*np.array(s) for s in [[.5,.25,.25],[.8,.1,.1],[.1,.45,.45]]]
                fits=[fit(f,s,gtol=1e-9) for s in starts]
                best=min(fits,key=lambda o:o[1].value)
                mu,_,_=f.prediction(best[0],best[1])
                loss=float(np.mean((f.test[1]-mu)**2))
                target=old[(old.seed==run.seed)&(old.model==m)&(old.group==group)].iloc[0]
                rows.append({'regime':run.regime,'seed':int(run.seed),'model':m,
                             'group':int(group),'training_only_loss':loss,
                             'warm_reference_loss':target.reference_loss,
                             'absolute_difference':abs(loss-target.reference_loss),
                             'restart_objective_spread':max(z[1].value for z in fits)-min(z[1].value for z in fits),
                             'inside_recorded_interval':target.lower-1e-8<=loss<=target.upper+1e-8})
        if len(rows)%360==0:
            print(f'{len(rows)} folds checked',flush=True)
    df=pd.DataFrame(rows);df.to_csv(root/'training_start_audit.csv',index=False)
    print(df[['absolute_difference','restart_objective_spread']].max().to_string(),flush=True)
    print('outside intervals',sum(~df.inside_recorded_interval),flush=True)


if __name__=='__main__':main()
