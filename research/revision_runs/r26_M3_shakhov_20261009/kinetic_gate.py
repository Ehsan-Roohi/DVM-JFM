"""Independent 3D Gaussian projection of Shakhov collision and Grad26 transport.

No closure formula is used to compute the quadrature oracle. Eq29 defines fG.
Complex-step differentiates f at FIXED laboratory velocities, not moving nodes.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,dataclasses
import numpy as np
from numpy.polynomial.hermite import hermgauss
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'physics'))
import implicit_shakhov as im
s=im.s
from r26_tensor_closures import stf2_project,stf3_project,stf4_project

def grid(T):
    h,w=hermgauss(9);C=np.array(np.meshgrid(h,h,h,indexing='ij')).reshape(3,-1).T*np.sqrt(2*T)
    W=np.einsum('i,j,k->ijk',w,w,w).ravel()/np.pi**1.5
    return C,W

def weights(C,T):
    C2=np.sum(C*C,axis=1);x=C[:,0]
    p2=x*x-C2/3;p3=x**3-3*x*C2/5;p4=x**4-6*x*x*C2/7+3*C2*C2/35
    return np.column_stack([np.ones(len(C)),x,C2,p2,.5*x*C2,p3,p2*(C2-7*T),C2*C2-15*T*T,p4,p3*(C2-9*T),x*C2*(C2-14*T)])

def poly(C,rho,T,q,sig,m,R,D,phi=None,psi=None,omega=None):
    C2=np.sum(C*C,axis=1);p=rho*T
    v=1+np.einsum('ij,ki,kj->k',sig,C,C)/(2*p*T)
    v+=C@q/(p*T)*(C2/(5*T)-1)
    v+=np.einsum('ijk,li,lj,lk->l',m,C,C,C)/(6*p*T*T)
    v+=np.einsum('ij,ki,kj->k',R,C,C)/(4*p*T*T)*(C2/(7*T)-1)
    v+=D/(8*p*T)*(C2*C2/(15*T*T)-2*C2/(3*T)+1)
    if phi is not None:v+=np.einsum('ijkl,ni,nj,nk,nl->n',phi,C,C,C,C)/(24*p*T**3)
    if psi is not None:v+=np.einsum('ijk,li,lj,lk->l',psi,C,C,C)/(12*p*T**3)*(C2/(9*T)-1)
    if omega is not None:v+=C@omega/(40*p*T*T)*(C2*C2/(7*T*T)-2*C2/T+5)
    return v

def state_from_z(p,z):
    v=z*p.scale;u=np.exp(v[0]);T=np.exp(v[1]);rho=p.mass/u
    sig=p.momentum-p.mass*u-rho*T;q=p.energy-p.mass*(u*u/2+2.5*T)-u*sig
    sigma=np.diag([sig,-sig/2,-sig/2]);rr=np.diag([v[3],-v[3]/2,-v[3]/2])
    mm=np.zeros((3,3,3),dtype=np.result_type(v));mm[0,0,0]=v[2]
    for j in [1,2]:mm[0,j,j]=mm[j,0,j]=mm[j,j,0]=-v[2]/2
    return rho,u,T,np.array([q,0,0]),sigma,mm,rr,v[4]

def run():
    rng=np.random.default_rng(773);report={'method':'9^3 Gauss-Hermite; independent Eq29 polynomial; fixed lab velocity complex step','collision':[],'closure_transport':[]}
    for j in range(12):
        rho,T=rng.uniform(.7,3,2);C,W=grid(T);P=weights(C,T)
        sig=stf2_project(rng.normal(size=(3,3)))*.07*rho*T
        m=stf3_project(rng.normal(size=(3,3,3)))*.05*rho*T**1.5
        R=stf2_project(rng.normal(size=(3,3)))*.1*rho*T*T
        q=rng.normal(size=3)*.04*rho*T**1.5;D=.12*rho*T*T
        phi=stf4_project(rng.normal(size=(3,3,3,3)))*.03*rho*T*T
        psi=stf3_project(rng.normal(size=(3,3,3)))*.02*rho*T**2.5;omega=rng.normal(size=3)*.03*rho*T**2.5
        G=poly(C,rho,T,q,sig,m,R,D,phi,psi,omega)
        C2=np.sum(C*C,axis=1);S=1+(1-2/3)*(C@q)/(5*rho*T*T)*(C2/T-5)
        actual=rho*rho*(P.T@(W*(S-G)))
        expected=np.r_[0,0,0,-rho*sig[0,0],-2*rho*q[0]/3,-rho*m[0,0,0],-rho*R[0,0],-rho*D,-rho*phi[0,0,0,0],-rho*psi[0,0,0],-rho*omega[0]]
        err=float(np.max(abs(actual-expected)/np.maximum(1,abs(expected))))
        report['collision'].append({'case':j,'max_scaled_error':err})
    p=s.Shock(7681,45,3)
    a=np.load(OUT.parent/'r26_M3_discretization_repair_20261008/mixed7681/profile.npz')
    dz=np.gradient(a['z'],a['x'],axis=0,edge_order=2)
    for x in [-9,-6.7,-3,-1.8,0,1,6.13]:
        idx=np.argmin(abs(a['x']-x));z=a['z'][idx];zp=dz[idx]
        rho,u,T,q,sig,m,R,D=state_from_z(p,z);C,W=grid(T);lab=C.copy();lab[:,0]+=u
        baseC2=np.sum(C*C,axis=1);P=weights(C,T)
        def f_ratio(zz):
            rr,uu,TT,qq,ss,mm,RR,DD=state_from_z(p,zz)
            CC=lab.astype(np.result_type(zz)).copy();CC[:,0]-=uu
            return rr/rho*(T/TT)**1.5*np.exp(baseC2/(2*T)-np.sum(CC*CC,axis=1)/(2*TT))*poly(CC,rr,TT,qq,ss,mm,RR,DD)
        fd=np.imag(f_ratio(z+1j*1e-20*zp))/1e-20
        transport=rho*(P.T@(W*lab[:,0]*fd))
        expected=-transport[8:]/rho
        Y=np.r_[z,np.zeros(3)][None,:];Yp=np.r_[zp,np.zeros(3)][None,:]
        _,cl,_,_=im.point_rows(p,Y,Yp,details=True)
        err=(cl[0]-expected)/np.maximum(1,abs(expected))
        report['closure_transport'].append({'x':float(a['x'][idx]),'quadrature':expected.tolist(),'kernel':cl[0].tolist(),'scaled_difference':err.tolist()})
    report['collision_max']=max(x['max_scaled_error'] for x in report['collision'])
    report['closure_transport_max']=max(np.max(np.abs(x['scaled_difference'])) for x in report['closure_transport'])
    report['status']='PASS' if report['collision_max']<1e-10 and report['closure_transport_max']<1e-9 else 'FAIL_DIAGNOSE'
    (OUT/'kinetic_gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
