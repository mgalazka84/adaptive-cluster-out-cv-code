"""Exclusive phase timing, separate from randomized benchmark timings.

Fixed sample: first five seeds of four development settings, five larger
fresh datasets, and both real-data targets; three profiling repetitions.
The profiler itself adds overhead, so these are cost explanations, not
replacement speed comparisons. Nested work belongs to its outermost phase.
"""
from pathlib import Path
from collections import defaultdict
from functools import wraps
from time import perf_counter
import pandas as pd
import boundary_acv as bc
from acv import CompressedFold
from run_accuracy_benchmark import run,problem
from run_lexdec import make_problem
from accuracy_io import read_csv
BASE=Path(__file__).resolve().parents[1]/'results'/'accuracy_v2'
phase=None;seconds=defaultdict(float);calls=defaultdict(int);allcalls=defaultdict(int);mode=None
def wrap(owner,name,label):
 original=getattr(owner,name)
 @wraps(original)
 def timed(*args,**kwargs):
  global phase
  allcalls[name]+=1
  if phase is not None or (name=='fit_box' and mode!='adaptive'):
   return original(*args,**kwargs)
  phase=label;t=perf_counter()
  try:return original(*args,**kwargs)
  finally:seconds[label]+=perf_counter()-t;calls[label]+=1;phase=None
 setattr(owner,name,timed)
for name,label in [('active_step','steps'),('face_certificate','certificates'),('same_root','anchors'),('fit_box','fallback')]:wrap(bc,name,label)
for name,label in [('evaluate','evaluations'),('value_gradient','gradient_evaluations'),('objective','objectives'),('prediction','prediction')]:wrap(CompressedFold,name,label)
# The benchmark module imports functions by name; update those aliases too.
import run_accuracy_benchmark as harness
for name in ['active_step','face_certificate','fit_box']:setattr(harness,name,getattr(bc,name))
jobs=[]
r=read_csv(BASE/'original'/'runs.csv')
for regime in ['balanced','sparse','boundary','zero_both']:jobs.extend(r[r.regime==regime].head(5).to_dict('records'))
r=read_csv(BASE/'fresh'/'runs.csv');jobs.extend(r[r.regime=='many_levels'].head(5).to_dict('records'))
jobs.extend(read_csv(BASE/'lexdec'/'runs.csv').to_dict('records'))
rows=[]
for job in jobs:
 if job['suite']=='lexdec':folds,starts,_,_=make_problem(job['regime'])
 else:folds,starts,_=problem(job['regime'],int(job['seed']),*( (30,24,8) if job['regime']=='many_levels' else (12,10,3)))
 for method in ['newton','practical','adaptive']:
  mode=method
  for rep in range(3):
   seconds.clear();calls.clear();allcalls.clear()
   t=perf_counter();result=run(method,folds,starts);total=perf_counter()-t
   row=dict(suite=job['suite'],regime=job['regime'],seed=job['seed'],method=method,repeat=rep,total_seconds=total,
       other_seconds=total-sum(seconds.values()),iterations=sum(result['iterations']),tests=sum(result['checks']))
   row.update({k+'_seconds':v for k,v in seconds.items()});row.update({k+'_calls':v for k,v in allcalls.items()});rows.append(row)
 print(job['suite'],job['regime'],job['seed'],flush=True)
pd.DataFrame(rows).fillna(0).to_csv(BASE/'summary'/'cost_profiles.csv',index=False)
