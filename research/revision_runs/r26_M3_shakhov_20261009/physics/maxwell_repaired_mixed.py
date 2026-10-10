"""Printed Maxwell R26 with three STF bracket repairs; independent scalar projection.

Differs explicitly from copied Gu nonlinear transport; not a silent patch.
Original Maxwell molecule collision coefficients and nonlinear productions retained.
Pr2/3, mu=T and RH/phase constraints are fixed.
"""
import numpy as np
import shock_pilot as s
import projected_transport as pt
from implicit_shakhov import sparsity
Shock=s.Shock

def values(p,Y,Yp):
    Y=np.asarray(Y).reshape(-1,8);Yp=np.asarray(Yp).reshape(-1,8)
    v=Y[:,:5]*p.scale;dv=Yp[:,:5]*p.scale
    u=np.exp(v[:,0]);T=np.exp(v[:,1]);rho=p.mass/u
    sig=p.momentum-p.mass*u-rho*T
    q=p.energy-p.mass*(u*u/2+2.5*T)-u*sig
    return rho,u,T,sig,q,v[:,2],v[:,3],v[:,4],u*dv[:,0],T*dv[:,1],dv[:,2],dv[:,3],dv[:,4]

def point_rows(p,Y,Yp,details=False):
    Y=np.asarray(Y).reshape(-1,8);Yp=np.asarray(Yp).reshape(-1,8)
    rho,u,T,sig,q,m,R,D,du,dT,dm,dR,dD=values(p,Y,Yp)
    calc=pt.constitutive(rho,u,T,sig,q,m,R,D,du,dT,dm,dR,dD)
    calc[:,0]=(calc[:,0]-.291*0.5142857142857142*sig*sig/rho)/2.097
    calc[:,1]=(calc[:,1]-1.203*0.4*sig*m/rho-.854*0.6*q*sig/rho)/1.698
    calc[:,2]-=2*(14*q*sig+7.5*m*sig)/(15*rho)
    raw=pt.transport(rho,u,T,sig,q,m,R,D,*Y[:,5:].T,du,dT,dm,dR,dD,*Yp[:,5:].T)
    rates=np.column_stack([rho*sig,2*rho*q/3,1.5*rho*m,7*rho*R/6+sig*sig/3,2*rho*D/3+sig*sig])
    bulk=raw+rates
    eq=np.column_stack([bulk[:,:2],calc-Y[:,5:],bulk[:,2:]])
    if details:return eq,calc,bulk
    return eq

def lift(p,z):
    z=np.asarray(z).reshape(p.n,5)
    Y=np.column_stack([z,np.zeros((p.n,3))]);Yp=np.column_stack([np.gradient(z,p.x,axis=0,edge_order=2),np.zeros((p.n,3))])
    _,Y[:,5:],_=point_rows(p,Y,Yp,details=True)
    return Y

def residual_diagnostic(p,Y):
    Y=np.asarray(Y).reshape(p.n,8)
    scales=np.r_[p.eqscale[:2],[p.u1**2,p.u1**3,p.u1**3],p.eqscale[2:]]
    r=point_rows(p,(Y[1:]+Y[:-1])/2,np.diff(Y,axis=0)/np.diff(p.x)[:,None])/scales
    bc=np.r_[Y[0,:5]-p.left/p.scale,Y[-1,:5]-p.right/p.scale]
    phase=Y[p.n//2,0]-np.log(2*p.u1/(1+p.u1/p.u2))
    return np.r_[r.ravel(),100*bc,100*phase]

def nodal_rows(p,Y):
    """Own full nodal PDE with constitutive derivatives reconstructed separately."""
    Y=np.asarray(Y).reshape(p.n,8).copy();Y[:]=lift(p,Y[:,:5])
    Yp=np.gradient(Y,p.x,axis=0,edge_order=2)
    return point_rows(p,Y,Yp,details=True)[2]
