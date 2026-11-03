"""Dense reference for arbitrary fixed PSD variance components.

Independent random slopes are represented by separate Z_j Z_j.T components.
An unstructured correlated covariance is not covered by this parameterization.
"""
import numpy as np
from scipy.linalg import qr, cholesky, cho_solve, solve_triangular
from acv import Fold

class LinearPSDFold(Fold):
    def __init__(self, X, y, Z, test):
        self.X=np.asarray(X,float);self.y=np.asarray(y,float)
        self.n,self.p=self.X.shape
        if np.linalg.matrix_rank(self.X)!=self.p:raise ValueError('Rank deficient X')
        xs,ys,Zs=test
        self.test=(xs,ys,None,None)
        self.A=np.array([np.eye(self.n)]+[z@z.T for z in Z])
        self.cross=np.array([np.zeros((len(ys),self.n))]+[zs@z.T for zs,z in zip(Zs,Z)])
        self.K=qr(self.X,mode='full',check_finite=False)[0][:,self.p:]
        self.C=np.array([self.K.T@a@self.K for a in self.A])
        self.z=self.K.T@self.y;self.evaluations=0

    def prediction(self,theta,ev=None):
        if ev is None:ev=self.evaluate(theta)
        if ev.W is None:ev.W=solve_triangular(ev.chol,self.K.T,lower=True,check_finite=False).T
        v=np.einsum('i,ijk->jk',theta,self.A)
        l=cholesky(v,lower=True,check_finite=False)
        vx=cho_solve((l,True),self.X,check_finite=False)
        Q=np.linalg.solve(self.X.T@vx,vx.T)
        B=self.test[0]@Q+np.einsum('i,ijk->jk',theta,self.cross)@(ev.W@ev.W.T)
        return B@self.y,B,self.cross
