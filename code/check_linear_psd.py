"""Independent formula checks for the five-component reference implementation."""
import json
import numpy as np
from scipy.linalg import cho_solve
from run_revision_study import slope_problem,OUT

def main():
    folds,_=slope_problem(93005000);rng=np.random.default_rng(93009000);errors=[]
    for f in [folds[0][0],folds[1][4]]:
        t=np.array([1.,.4,.3,.15,.1]);mu,B,cross=f.prediction(t)
        for _ in range(8):
            delta=rng.normal(size=5)*.01;ev=f.evaluate(t+delta)
            new=f.prediction(t+delta,ev)[0]
            P=f.K@cho_solve((ev.chol,True),f.K.T)
            rhs=(np.einsum('i,ijk->jk',delta,cross)-B@np.einsum('i,ijk->jk',delta,f.A))@P@f.y
            errors.append(float(np.max(np.abs(new-mu-rhs))))
    assert max(errors)<1e-10
    result={'prediction_identity_checks':len(errors),'max_error':max(errors),'status':'passed',
      'scope':'Numerical regression test, not a proof or verified arithmetic.'}
    (OUT/'linear_psd_checks.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
