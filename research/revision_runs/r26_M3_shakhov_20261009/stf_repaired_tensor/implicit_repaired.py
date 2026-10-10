"""Implicit eight-state adjacent-node residual; diagnostic constraints, no solver.

Y=(production z[5], phi_xxxx, psi_xxx, Omega_x), last three in physical units.
Never forms A inverse. Point rows retain exact frozen R26 kernels.
"""
import sys, dataclasses
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import shock_pilot as s
from r26_tensor_closures import stf2_project,stf3_project,stf4_project


def point_rows(p,Y,Yprime,details=False):
    Y=np.atleast_2d(np.asarray(Y,dtype=float));Yprime=np.atleast_2d(np.asarray(Yprime,dtype=float));assert Y.shape==Yprime.shape and Y.shape[1]==8
    v=Y[:,:5]*p.scale;dv=Yprime[:,:5]*p.scale;u=np.exp(v[:,0]);T=np.exp(v[:,1]);rho=p.mass/u;sig=p.momentum-p.mass*u-rho*T;q=p.energy-p.mass*(u*u/2+2.5*T)-u*sig
    t=s.tensors_from_components(rho,u,T,q,sig,v[:,2],v[:,3],v[:,4]);du=u*dv[:,0];dT=T*dv[:,1];dr=-rho*dv[:,0];ds=-p.mass*du-dr*T-rho*dT;dq=-p.mass*(u*du+2.5*dT)-du*sig-u*ds
    dt=s.tensors_from_components(dr,du,dT,dq,ds,dv[:,2],dv[:,3],dv[:,4]);g=lambda a:np.stack([a,np.zeros_like(a),np.zeros_like(a)],axis=1)
    grad=s.R26Gradients(**{f.name:g(getattr(dt,f.name)) for f in dataclasses.fields(s.R26Gradients)})
    calc=s.closures_from_tensors(t,grad,mu=T,coefficient_mode='jfm2009');n=len(Y)
    phi=np.zeros((n,3,3,3,3));phi[:,0,0,0,0]=Y[:,5]*35/8;phi=stf4_project(phi)
    psi=np.zeros((n,3,3,3));psi[:,0,0,0]=Y[:,6]*5/2;psi=stf3_project(psi)
    omega=np.zeros((n,3));omega[:,0]=Y[:,7]
    independent=dataclasses.replace(calc,phi=phi,psi=psi,Omega=omega)
    dp=np.zeros((n,3,3,3));dp[:,0,0,0]=Yprime[:,5]*5/2;dp=stf3_project(dp)
    dpsi=np.zeros((n,3,3));dpsi[:,0,0]=Yprime[:,6]*3/2;dpsi=stf2_project(dpsi)
    domega=np.zeros((n,3,3));domega[:,0,0]=Yprime[:,7]
    r=s.steady_r26_bulk_residual(t,grad,independent,s.R26ClosureDerivatives(dp,dpsi,domega),mu=T)
    computed=np.column_stack([calc.phi[:,0,0,0,0],calc.psi[:,0,0,0],calc.Omega[:,0]])
    eq=np.column_stack([r.stress[:,0,0],r.heat_flux[:,0],computed-Y[:,5:],r.m[:,0,0,0],r.R[:,0,0],r.Delta])
    if details:return eq,computed,t,r
    return eq


def lift(p,z):
    """Construct independent closure components using original nodal D1; no solve."""
    z=np.asarray(z).reshape(p.n,5);Y=np.column_stack([z,np.zeros((p.n,3))]);dY=np.column_stack([np.gradient(z,p.x,axis=0,edge_order=2),np.zeros((p.n,3))]);_,C,_,_=point_rows(p,Y,dY,details=True);Y[:,5:]=C
    return Y


def midpoint_rows(p,Y):
    Y=np.asarray(Y).reshape(p.n,8);mid=(Y[1:]+Y[:-1])/2;deriv=np.diff(Y,axis=0)/np.diff(p.x)[:,None]
    return point_rows(p,mid,deriv)


def residual_diagnostic(p,Y):
    """Overdetermined diagnostic only: retain original10 endpoint+1phase rows.

8N unknowns, 8(N-1)+11=8N+3 residuals; no invented boundary deletion.
Row scales explicit only for conditioning; raw point/midpoint rows available.
"""
    Y=np.asarray(Y).reshape(p.n,8)
    # Original five PDE scales and explicit constitutive scales. No acceptance change.
    scales=np.r_[p.eqscale[:2],[p.u1**2,p.u1**3,p.u1**3],p.eqscale[2:]]
    rows=midpoint_rows(p,Y)/scales
    bc=np.r_[Y[0,:5]-p.left/p.scale,Y[-1,:5]-p.right/p.scale]
    umid=2*p.u1/(1+p.u1/p.u2);phase=Y[p.n//2,0]-np.log(umid)
    return np.r_[rows.ravel(),100*bc,100*phase]


def sparsity(p):
    from scipy.sparse import lil_matrix
    A=lil_matrix((8*(p.n-1)+11,8*p.n),dtype=int)
    for i in range(p.n-1):A[8*i:8*i+8,8*i:8*i+16]=1
    start=8*(p.n-1)
    for j in range(5):A[start+j,j]=1;A[start+5+j,8*(p.n-1)+j]=1
    A[-1,8*(p.n//2)]=1
    return A.tocsr()
