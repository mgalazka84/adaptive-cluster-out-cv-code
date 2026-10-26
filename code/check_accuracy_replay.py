"""Replay default adaptive decisions after adding persistent root anchors.

Checks every archived final interval and fit against released code; timings
are not repeated. A trace records unsuccessful tests after a valid anchor.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from accuracy_io import read_csv
import boundary_acv as bc
from run_accuracy_benchmark import problem
from run_lexdec import make_problem
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results'/'accuracy'
original=bc.State.refine
trace={'failed_test_after_valid_anchor':0,'valid_refinements':0}
def traced(self):
 old=self.anchor
 original(self)
 if old is not None:
  trace['valid_refinements']+=1
  if not self.cert.get('valid') and not self.numerical:trace['failed_test_after_valid_anchor']+=1
bc.State.refine=traced
count=0;max_difference=0.;rows=[]
for suite in ['original','scale','fresh','lexdec']:
 runs=read_csv(BASE/suite/'runs.csv');aud=read_csv(BASE/suite/'fold_audit.csv')
 for _,row in runs.iterrows():
  if suite=='lexdec':folds,starts,_,_=make_problem(row.regime)
  else:
   na,nb,M=(24,20,3) if suite=='scale' else ((30,24,8) if row.regime=='many_levels' else (12,10,3))
   folds,starts,_=problem(row.regime,int(row.seed),na,nb,M)
  best,states,lo,hi,rounds=bc.adaptive(folds,starts)
  archived=aud[(aud.regime==row.regime)&(aud.seed==row.seed)].sort_values(['model','group_index'])
  assert best==row.adaptive_winner and rounds==row.refinement_rounds
  assert sum(s.numerical for s in states)==row.numerical_fallbacks
  assert [s.steps for s in states]==archived.steps.tolist()
  assert [s.checks for s in states]==archived.checks.tolist()
  before=archived[['lower','upper']].to_numpy();after=np.array([[s.cert['lo'],s.cert['hi']] for s in states])
  delta=float(abs(before-after).max());max_difference=max(max_difference,delta)
  assert np.allclose(before,after,rtol=1e-13,atol=1e-13)
  count+=1
  rows.append(dict(suite=suite,regime=row.regime,seed=int(row.seed),max_interval_difference=delta))
 print(suite,count,trace,flush=True)
result=dict(runs=count,max_interval_difference=max_difference,**trace)
(BASE/'summary'/'replay_checks.json').write_text(json.dumps(result,indent=2))
pd.DataFrame(rows).to_csv(BASE/'summary'/'replay_checks.csv',index=False)
print(result)
