"""Independent numerical checks for analytic identities, not a theorem proof."""
import json
from pathlib import Path
import numpy as np
from acv import Fold, CompressedFold, simulate, make_fold, fit, newton_step, certificate


def main():
    errors={'gradient':[],'hessian':[],'prediction_identity':[],
            'gradient_kernel':[],'compression_objective':[],
            'compression_gradient':[],'compression_hessian':[],'compression_prediction':[]}
    local_checks=0; loss_checks=0
    for seed in range(6):
        X,y,a,b=simulate(500+seed,na=7,nb=6)
        f=make_fold(X,y,a,b,[0,1,2],seed%7)
        theta=np.array([1.1,.7,.5]);ev=f.evaluate(theta)
        fc=make_fold(X,y,a,b,[0,1,2],seed%7,'compressed');ec=fc.evaluate(theta)
        errors['compression_objective'].append(abs(ev.value-ec.value))
        errors['compression_gradient'].append(float(np.max(np.abs(ev.gradient-ec.gradient))))
        errors['compression_hessian'].append(float(np.max(np.abs(ev.hessian-ec.hessian))))
        errors['compression_prediction'].append(float(np.max(np.abs(f.prediction(theta,ev)[0]-fc.prediction(theta,ec)[0]))))
        step=1e-5
        numeric_g=[];numeric_H=[]
        for j in range(3):
            d=np.eye(3)[j]*step
            numeric_g.append((f.objective(theta+d)-f.objective(theta-d))/(2*step))
            numeric_H.append((f.evaluate(theta+d).gradient-
                              f.evaluate(theta-d).gradient)/(2*step))
        errors['gradient'].append(float(np.max(np.abs(ev.gradient-numeric_g))))
        errors['hessian'].append(float(np.max(np.abs(ev.hessian-np.array(numeric_H).T))))
        errors['gradient_kernel'].append(float(np.max(np.abs(ev.gradient-f.value_gradient(theta)[1]))))
        mu,B,cross=f.prediction(theta,ev)
        rng=np.random.default_rng(seed)
        for _ in range(8):
            delta=rng.normal(scale=.025,size=3)
            ev2=f.evaluate(theta+delta);mu2,_,_=f.prediction(theta+delta,ev2)
            dV=np.einsum('i,iab->ab',delta,f.A)
            dC=np.einsum('i,iab->ab',delta,cross)
            alpha2=ev2.W@ev2.w
            rhs=(dC-B@dV)@alpha2
            errors['prediction_identity'].append(float(np.max(np.abs(mu2-mu-rhs))))
        center=theta.copy()
        ref,er,_=fit(f,theta,gtol=1e-10)
        for k in range(8):
            e=f.evaluate(center);c=certificate(f,center,e)
            if c['valid'] and c['radius']>1e-7:
                dist=np.linalg.norm(np.linalg.solve(c['S'],ref-center))
                assert dist <= c['radius']+1e-7,(dist,c['radius'])
                local_checks+=1
                mse=np.mean((f.test[1]-f.prediction(ref,er)[0])**2)
                assert c['lo']-1e-8 <= mse <= c['hi']+1e-8,(mse,c)
                loss_checks+=1
            center=newton_step(f,center,e)
    maxima={key:max(vals) for key,vals in errors.items()}
    assert maxima['gradient']<1e-5,maxima
    assert maxima['hessian']<1e-4,maxima
    assert maxima['prediction_identity']<1e-9,maxima
    assert maxima['gradient_kernel']<1e-9,maxima
    for key in ['compression_objective','compression_gradient','compression_hessian','compression_prediction']:
        assert maxima[key]<1e-9,maxima
    assert local_checks>0
    # Identical models must produce an unresolved tie, not a unique winner.
    from run_pilot import adaptive
    X,y,a,b=simulate(991,na=5,nb=4)
    full=CompressedFold(X[:,:3],y,a,b)
    theta,_,_=fit(full,np.array([1.,.7,.5]))
    folds=[make_fold(X,y,a,b,[0,1,2],k,'compressed') for k in np.unique(a)]
    winner,*_=adaptive([folds,folds],[theta,theta],len(folds))
    assert winner is None,'Identical models must remain tied'
    report={'max_absolute_errors':maxima,'gradient_checks':18,'hessian_entries':54,
            'prediction_identity_checks':48,'local_root_checks':local_checks,
            'loss_enclosure_checks':loss_checks,'compression_instances':6,
            'identical_model_tie_checks':1,'status':'passed',
            'warning':'Numerical tests do not prove the theorems or directed-rounding validity.'}
    path=Path(__file__).resolve().parents[1]/'results'/'math_checks.json'
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
