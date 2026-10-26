"""80-digit constants and gradient audit; not directed-rounding verification.

Constants use binary64 whitened inputs. The gradient is additionally rebuilt
with an 80-digit Cholesky factorization of the compressed covariance. QR and
the compressed source arrays, and S used to measure eta, remain binary64.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from accuracy_io import read_csv
import mpmath as mp
from run_accuracy_benchmark import problem
from run_lexdec import make_problem
from boundary_acv import face_certificate
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results'/'accuracy_v2'
mp.mp.dps=80

def matrix(a):return mp.matrix([[mp.mpf(float(x)) for x in row] for row in np.asarray(a)])
def vec(a):return mp.matrix([mp.mpf(float(x)) for x in a])
def eigmax(a):return max(mp.eigsy(a,eigvals_only=True))
def sym(a):return (a+a.T)/2

def gradient_audit(fold,theta,ev,cert):
 cs=[matrix(c) for c in fold.C];th=vec(theta);z=vec(fold.z)
 C=mp.zeros(len(z))
 for t,c in zip(th,cs):C+=t*c
 L=mp.cholesky(C);n=len(z);Li=mp.zeros(n)
 for j in range(n):
  for i in range(j,n):
   Li[i,j]=((1 if i==j else 0)-sum(L[i,k]*Li[k,j] for k in range(j,i)))/L[i,i]
 inv=Li.T*Li;alpha=inv*z
 g=mp.matrix([mp.mpf('.5')*(sum(inv[i,j]*c[j,i] for i in range(n) for j in range(n))-(alpha.T*c*alpha)[0]) for c in cs])
 g[0]+=mp.mpf('.5')*fold.d0/th[0]-mp.mpf('.5')*mp.mpf(fold.s0)/th[0]**2
 S=matrix(cert['S']);eta=mp.norm(S.T*g)
 gd=vec(ev.gradient);eta_double=mp.mpf(cert['eta'])
 out=dict(gradient_max_absolute_error=float(max(abs(a-b) for a,b in zip(g,gd))),
  gradient_scaled_error=float(mp.norm(S.T*(g-gd))),eta_double=float(eta_double),eta_80digit=float(eta),
  eta_absolute_error=float(abs(eta-eta_double)),eta_relative_error=float(abs(eta-eta_double)/max(eta,mp.mpf('1e-300'))),
  eta_underestimated=bool(eta_double<eta))
 for j in range(3):out['gradient_double_'+str(j)]=float(gd[j]);out['gradient_80digit_'+str(j)]=float(g[j])
 return out

def constants(fold,theta):
 ev=fold.evaluate(theta);cert=face_certificate(fold,theta,ev);assert cert['valid']
 S=cert['S'];D=np.einsum('ij,iab->jab',S,ev.b);extra=ev.extra_b@S if ev.extra_dim else np.zeros(S.shape[1]);_,G=fold.prediction_geometry(theta,ev,S)
 dm=[matrix(d) for d in D];w=vec(ev.w);ex=vec(extra);w2=mp.mpf(ev.extra_w2);h=ev.extra_dim
 ops=[max(abs(x) for x in mp.eigsy(sym(d),eigvals_only=True)) for d in dm]
 a=mp.sqrt(sum(max(o,abs(e))**2 for o,e in zip(ops,extra)))
 f2=sum(x*x for d in dm for x in d)+h*sum(x*x for x in ex)
 dw=mp.matrix(len(w),len(dm))
 for j,d in enumerate(dm):dw[:,j]=d*w
 b0=mp.sqrt(max(mp.mpf(0),eigmax(sym(dw.T*dw+w2*(ex*ex.T)))))
 wn=mp.sqrt((w.T*w)[0]+w2);r=mp.mpf(cert['outer_radius']);b=b0+a*a*r/(1-a*r)*wn
 lr=(f2*a/(1-a*r)**3+3*b*b*a/(1-a*r)**2)*r
 gm=[matrix(g) for g in G];c=mp.sqrt(sum(x*x for g in gm for x in g));gw=mp.matrix(G.shape[1],len(gm))
 for j,g in enumerate(gm):gw[:,j]=g*w
 c0=mp.sqrt(max(mp.mpf(0),eigmax(sym(gw.T*gw))))
 values={'a':a,'f2':f2,'b0':b0,'b_r':b,'c':c,'c0':c0,'Lr':lr}
 return {key+'_relative_difference':float(abs(val-mp.mpf(cert[key]))/max(abs(val),mp.mpf('1e-300'))) for key,val in values.items()}|{'contraction_margin':1-cert['Lr']}|gradient_audit(fold,theta,ev,cert)

def main():
 frames={s:read_csv(BASE/s/'fold_audit.csv') for s in ['original','scale','fresh','lexdec']};chosen=[]
 for active in [0,1,2]:
  d=frames['original'];d=d[d.local_valid & (d.active_count==active)];chosen.append(d.loc[d.Lr.idxmax()])
 for suite,regime in [('scale','balanced'),('fresh','many_levels'),('lexdec','new_subject'),('lexdec','new_word')]:
  d=frames[suite];d=d[d.local_valid & (d.regime==regime)];chosen.append(d.loc[d.Lr.idxmax()])
 for active in [0,1,2]:
  d=frames['original'];d=d[d.local_valid & (d.active_count==active)];chosen.append(d.loc[d.eta.idxmin()])
 for regime in ['new_subject','new_word']:
  d=frames['lexdec'];d=d[d.local_valid & (d.regime==regime)];chosen.append(d.loc[d.eta.idxmin()])
 allfolds=pd.concat(frames.values(),ignore_index=True)
 chosen.append(allfolds.loc[allfolds.parameter_ratio.idxmin()])
 chosen=list({(r.suite,r.regime,int(r.seed),int(r.model),int(r.group_index)):r for r in chosen}.values())
 rows=[]
 for row in chosen:
  if row.suite=='lexdec':folds,_,_,_=make_problem(row.regime)
  else:
   na,nb,M=(24,20,3) if row.suite=='scale' else ((30,24,8) if row.regime=='many_levels' else (12,10,3))
   folds,_,_=problem(row.regime,int(row.seed),na,nb,M)
  result=constants(folds[int(row.model)][int(row.group_index)],np.array([row[f'theta_{j}'] for j in range(3)]))
  result.update(suite=row.suite,regime=row.regime,seed=int(row.seed),model=int(row.model),group_index=int(row.group_index),active_count=int(row.active_count))
  rows.append(result);print(result,flush=True)
  pd.DataFrame(rows).to_csv(BASE/'summary'/'precision_audit.csv',index=False)
if __name__=='__main__':main()
