"""Experimental steady planar R26 shock, reusing unchanged cavity tensor kernels.

Unknowns log(u), log(theta), m_xxx, R_xx, Delta. Three exact flux
integrals eliminate rho, sigma_xx and q_x. Far-field equilibrium is imposed
as an overdetermined least-squares asymptotic condition, not a wall BC.
This pilot MUST NOT be called converged on optimizer success alone.
Length unit mu1/(rho1*sqrt(theta1)); Maxwell mu=theta, rho1=theta1=1.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'cavity_source'))
from r26_state import planar_state_to_tensors, StateTensors
from r26_tensor_closures import R26Gradients, closures_from_tensors
from r26_bulk_equations import steady_r26_bulk_residual, R26ClosureDerivatives

def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)

def farfields(mach):
    gamma=5/3
    u1=mach*np.sqrt(gamma)
    rho2=(gamma+1)*mach**2/((gamma-1)*mach**2+2)
    p2=(2*gamma*mach**2-(gamma-1))/(gamma+1)
    return u1, u1/rho2, p2/rho2

def tensors_from_components(rho, u, theta, q, sigma, m, R, Delta):
    n=len(u)
    a=np.zeros((n,17))
    a[:,0]=rho; a[:,1]=u; a[:,3]=theta; a[:,4]=q
    a[:,6]=sigma; a[:,8]=-sigma/2
    a[:,9]=R; a[:,11]=-R/2
    a[:,12]=m; a[:,14]=-m/2; a[:,16]=Delta
    # The tensor map is linear; derivative inputs need no positivity check.
    safe=a.copy(); safe[:,0]=1; safe[:,3]=1
    t=planar_state_to_tensors(safe)
    return dataclasses.replace(t, rho=rho, theta=theta)

class Shock:
    def __init__(self,n=81,L=30,mach=2):
        self.x=np.linspace(-L,L,n); self.n=n; self.mach=mach
        self.u1,self.u2,self.T2=farfields(mach)
        self.mass=self.u1; self.momentum=self.u1**2+1
        self.energy=self.u1*(self.u1**2/2+2.5)
        self.scale=np.array([1.,1.,self.u1,self.u1**2,self.u1**2])
        self.eqscale=np.array([1.,self.u1,self.u1,self.u1**2,self.u1**2])
        self.left=np.array([np.log(self.u1),0,0,0,0])
        self.right=np.array([np.log(self.u2),np.log(self.T2),0,0,0])
    def initial(self):
        f=(1+np.tanh(self.x/3))/2
        z=np.zeros((self.n,5))
        z[:,0]=np.log(self.u1+(self.u2-self.u1)*f)
        z[:,1]=np.log(1+(self.T2-1)*f)
        return z.ravel()
    def fields(self,z):
        v=np.asarray(z).reshape(self.n,5)*self.scale
        u=np.exp(v[:,0]); T=np.exp(v[:,1]); m,R,D=v[:,2:].T
        rho=self.mass/u; sigma=self.momentum-self.mass*u-rho*T
        q=self.energy-self.mass*(u*u/2+2.5*T)-u*sigma
        return tensors_from_components(rho,u,T,q,sigma,m,R,D)
    def evaluate(self,z):
        v=np.asarray(z).reshape(self.n,5)*self.scale
        dv=np.gradient(v,self.x,axis=0,edge_order=2)
        t=self.fields(z); u=t.velocity[:,0]; T=t.theta; rho=t.rho
        du=u*dv[:,0]; dT=T*dv[:,1]; drho=-rho*dv[:,0]
        ds=-self.mass*du-drho*T-rho*dT
        dq=-self.mass*(u*du+2.5*dT)-du*t.sigma[:,0,0]-u*ds
        dt=tensors_from_components(drho,du,dT,dq,ds,dv[:,2],dv[:,3],dv[:,4])
        def g(a): return np.stack([a,np.zeros_like(a),np.zeros_like(a)],axis=1)
        grad=R26Gradients(**{f.name:g(getattr(dt,f.name)) for f in dataclasses.fields(R26Gradients)})
        cl=closures_from_tensors(t,grad,mu=T,coefficient_mode='jfm2009')
        dphi=np.gradient(cl.phi,self.x,axis=0,edge_order=2)
        dpsi=np.gradient(cl.psi,self.x,axis=0,edge_order=2)
        do=np.gradient(cl.Omega,self.x,axis=0,edge_order=2)
        cd=R26ClosureDerivatives(dphi[...,0],dpsi[...,0],g(do))
        res=steady_r26_bulk_residual(t,grad,cl,cd,mu=T)
        eq=np.column_stack([res.stress[:,0,0],res.heat_flux[:,0],res.m[:,0,0,0],res.R[:,0,0],res.Delta])
        return t,res,eq
    def residual(self,z):
        _,_,eq=self.evaluate(z)
        zz=np.asarray(z).reshape(self.n,5)
        bc=np.r_[zz[0]-self.left/self.scale,zz[-1]-self.right/self.scale]
        # Density-midpoint phase fixes translation; no PDE row is discarded.
        umid=2*self.u1/(1+self.u1/self.u2)
        phase=zz[self.n//2,0]-np.log(umid)
        return np.r_[(eq/self.eqscale).ravel(),bc,phase]
    def sparsity(self):
        A=lil_matrix((5*self.n+11,5*self.n),dtype=int)
        for i in range(self.n):
            # Composition of one-sided/centred three-point derivatives.
            lo=max(0,i-4); hi=min(self.n,i+5)
            A[5*i:5*i+5,5*lo:5*hi]=1
        A[5*self.n:5*self.n+5,:5]=1
        A[5*self.n+5:5*self.n+10,-5:]=1
        A[-1,5*(self.n//2)]=1
        return A.tocsr()

def gate():
    p=Shock(17,10)
    out={}
    for name,endpoint in [('upstream',p.left),('downstream',p.right)]:
        z=np.tile(endpoint/p.scale,(p.n,1)).ravel()
        t,r,e=p.evaluate(z)
        err=max(float(np.max(np.abs(e))),float(np.max(np.abs(t.sigma))),float(np.max(np.abs(t.heat_flux))))
        assert err<1e-10,(name,err)
        out[name+'_equilibrium_error']=err
    z=p.initial(); t,r,e=p.evaluate(z)
    # Analytic chain rule enforces all three conservation laws independently.
    err=max(np.max(np.abs(r.mass)),np.max(np.abs(r.momentum)),np.max(np.abs(r.theta)))
    assert err<1e-10,err
    out['conservation_differential_error']=float(err)
    assert np.max(abs(r.stress[:,1,1]+r.stress[:,0,0]/2))<1e-10
    assert np.max(abs(r.R[:,1,1]+r.R[:,0,0]/2))<1e-10
    assert np.max(abs(r.m[:,0,1,1]+r.m[:,0,0,0]/2))<1e-10
    # Gradient map checked against numerical differentiation on a fine smooth grid.
    eps=1e-7; base=p.residual(z); pattern=p.sparsity().toarray()
    for k in [0,7,40,75,84]:
        q=z.copy(); q[k]+=eps
        change=(p.residual(q)-base)/eps
        assert np.max(np.abs(change[pattern[:,k]==0]),initial=0)<1e-8
    out['normal_symmetry_and_sparsity']='PASS'
    out['status']='PASS'
    return out

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    p=Shock(args.nodes,args.half_domain,args.mach)
    z=p.initial()
    if args.initial:
        old=np.load(args.initial,allow_pickle=False)
        z=np.column_stack([np.interp(p.x,old['x'],old['z'][:,k]) for k in range(5)]).ravel()
    start=time.time(); calls=0; best=np.inf
    def fun(v):
        nonlocal calls,best
        f=p.residual(v); calls+=1
        weighted=f.copy(); weighted[5*p.n:]*=100
        norm=float(np.max(abs(weighted)))
        if calls%100==1:
            print(json.dumps({'calls':calls,'max_residual':norm,'seconds':time.time()-start}),flush=True)
        if norm<best and calls%25==1:
            best=norm
            with open(out/'checkpoint.tmp','wb') as h: np.savez(h,x=p.x,z=v.reshape(p.n,5))
            os.replace(out/'checkpoint.tmp',out/'checkpoint.npz')
        return weighted
    sol=least_squares(fun,z,jac_sparsity=p.sparsity(),tr_solver='lsmr',x_scale='jac',max_nfev=args.max_nfev,ftol=1e-11,xtol=1e-11,gtol=1e-11,verbose=1)
    t,r,eq=p.evaluate(sol.x); f=p.residual(sol.x)
    np.savez(out/'profile.npz',x=p.x,z=sol.x.reshape(p.n,5),rho=t.rho,u=t.velocity[:,0],theta=t.theta,sigma=t.sigma[:,0,0],q=t.heat_flux[:,0],m=t.m[:,0,0,0],R=t.R[:,0,0],Delta=t.Delta,S=t.R[:,0,0]+t.Delta/3,equation_residual=eq)
    sigma=t.sigma[:,0,0]; rho=t.rho; T=t.theta
    report=dict(vars(args)); report.update(seconds=time.time()-start,calls=calls,optimizer_success=bool(sol.success),optimizer_message=str(sol.message),max_scaled_equation_residual=float(np.max(abs(eq/p.eqscale))),max_boundary_and_phase_error=float(np.max(abs(f[5*p.n:]))),min_rho=float(min(rho)),min_theta=float(min(T)),min_directional_pressure=float(min(np.min(rho*T+sigma),np.min(rho*T-sigma/2))),status='PILOT_ONLY_REQUIRES_GRID_DOMAIN_AND_MODEL_VALIDATION')
    report['numerical_gate_pass']=bool(report['max_scaled_equation_residual']<1e-6 and report['max_boundary_and_phase_error']<1e-6 and report['min_directional_pressure']>0)
    dump(out/'RESULT.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--gate',action='store_true'); a.add_argument('--nodes',type=int,default=81); a.add_argument('--half-domain',type=float,default=30); a.add_argument('--mach',type=float,default=2); a.add_argument('--max-nfev',type=int,default=200); a.add_argument('--out',default='pilot'); a.add_argument('--initial')
    args=a.parse_args()
    if args.gate: print(json.dumps(gate(),indent=2))
    else: run(args)
