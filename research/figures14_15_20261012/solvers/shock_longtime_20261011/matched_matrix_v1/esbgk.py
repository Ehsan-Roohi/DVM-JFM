"""Reduced ES-BGK normal shock; exact transverse Gaussian integration.

Backward Euler collision + first-order conservative upwind transport.
Pr=2/3, alpha=-1/2, mu=T, tau_ES=(1-alpha)*mu/p.
Finite axial quadrature errors are measured, never silently normalized.
"""
import argparse, hashlib, json, os, signal, time
from pathlib import Path
import numpy as np
import shakhov_reference as ref
ROOT=Path(__file__).resolve().parent
STOP=False
def stop(*args):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop)
if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,stop)
def target(v,rho,u,T,sigma,alpha):
    tx=T+alpha*sigma/rho; ty=T-alpha*sigma/(2*rho)
    if not np.all(np.isfinite(tx+ty)) or np.min(tx)<=0 or np.min(ty)<=0:
        raise FloatingPointError('Non-positive ES target covariance')
    g=rho[:,None]/np.sqrt(2*np.pi*tx[:,None])*np.exp(-(v[None,:]-u[:,None])**2/(2*tx[:,None]))
    return np.stack([g,2*ty[:,None]*g,8*ty[:,None]**2*g])
def collision(f,v,w,c,dt):
    rho=np.sum(f[0]*w,axis=1);u=np.sum(f[0]*v*w,axis=1)/rho
    xx=np.sum(f[0]*v*v*w,axis=1)-rho*u*u
    T=(xx+np.sum(f[1]*w,axis=1))/(3*rho)
    if not np.all(np.isfinite(rho+u+T)) or np.min(rho)<=0 or np.min(T)<=0:
        raise FloatingPointError('Non-positive/non-finite gas state')
    alpha=1-1/c['Pr'];tau=(1-alpha)*c['mu1']*T**(c['omega']-1)/rho
    r=dt/tau
    sigma=(xx-rho*T)/(1+(1-alpha)*r)
    E=target(v,rho,u,T,sigma,alpha)
    return (f+r[None,:,None]*E)/(1+r[None,:,None])
def advance(f,v,w,left,right,c,dt):
    # One shared upwind interface flux per face; reservoir incoming values.
    face=np.where((v>=0)[None,None,:],np.concatenate([left[:,None,:],f],axis=1),np.concatenate([f,right[:,None,:]],axis=1))
    transported=f-dt/(2*c['L']/c['nx'])*v[None,None,:]*(face[:,1:]-face[:,:-1])
    if np.min(transported)<-1e-14: raise FloatingPointError('Negative reduced distribution')
    return collision(transported,v,w,c,dt)
def raw_invariants(f,v,w):
    return np.stack([np.sum(f[0]*w,axis=1),np.sum(f[0]*v*w,axis=1),.5*np.sum((f[0]*v*v+f[1])*w,axis=1)],axis=1)
def gate():
    c=dict(mach=3,nx=12,L=45,nv=801,vmax=40,Pr=2/3,mu1=1,omega=1,CFL=.8,tmax=400)
    x,v,w,f,left,right=ref.setup(c);dt=.037
    a=ref.moments(f,v,w);b=ref.moments(collision(f,v,w,c,dt),v,w)
    alpha=-.5;tau=1.5/a[:,0];r=dt/tau
    inv=np.max(abs(raw_invariants(collision(f,v,w,c,dt),v,w)-raw_invariants(f,v,w))/np.maximum(abs(raw_invariants(f,v,w)),1))
    stress=np.max(abs(b[:,4]-a[:,4]/(1+1.5*r)))
    heat=np.max(abs(b[:,3]-a[:,3]/(1+r)))
    # Exact BE scalar source evaluated at the implicit target stress.
    dg=3*alpha**2*b[:,4]**2/b[:,0]
    fourth=np.max(abs(b[:,6]-(a[:,6]+r*dg)/(1+r)))
    equilibrium=ref.eq(v,np.ones(12),np.full(12,3.),np.ones(12))
    eqerr=np.max(abs(advance(equilibrium,v,w,equilibrium[:,0],equilibrium[:,0],c,.001)-equilibrium))
    # Independent direct Cartesian quadrature of a non-equilibrium Gaussian.
    vt=np.linspace(-12,12,161);wt=np.full(161,vt[1]-vt[0]);wt[[0,-1]]*=.5
    sig=.4;tx=1+alpha*sig;ty=1-alpha*sig/2
    perp=np.exp(-(vt[:,None]**2+vt[None,:]**2)/(2*ty))/(2*np.pi*ty)
    rr=vt[:,None]**2+vt[None,:]**2;ww=wt[:,None]*wt[None,:]
    transverse=max(abs(np.sum(perp*ww)-1),abs(np.sum(perp*rr*ww)-2*ty),abs(np.sum(perp*rr**2*ww)-8*ty**2))
    # Coefficients must match mu/p, independently of collision update.
    viscosity=abs(tau[0]/(1-alpha)-1/a[0,0])
    assert inv<1e-11 and stress<1e-9 and heat<1e-9 and fourth<1e-8 and eqerr<1e-12 and transverse<1e-10 and viscosity<1e-14
    # Pure target at alpha=0 is Maxwellian, a distinct BGK-limit control.
    assert np.max(abs(target(v,a[:,0],a[:,1],a[:,2],a[:,4],0)-ref.eq(v,a[:,0],a[:,1],a[:,2])))<1e-12
    case_errors={}
    for case in json.loads((ROOT/'manifest.json').read_text())['cases']:
        cc=dict(case,nx=12);xx,vv,ww,ff,ll,rr=ref.setup(cc)
        ee=float(np.max(abs(raw_invariants(collision(ff,vv,ww,cc,.01),vv,ww)-raw_invariants(ff,vv,ww))/np.maximum(abs(raw_invariants(ff,vv,ww)),1)))
        assert ee<1e-10
        case_errors[case['name']]=ee
    f1=advance(f,v,w,left,right,c,.001);f2=advance(f1,v,w,left,right,c,.001)
    ref.save(ROOT/'gate_restart.npz',f=f1)
    loaded=np.load(ROOT/'gate_restart.npz')['f'];assert np.array_equal(f2,advance(loaded,v,w,left,right,c,.001))
    result=dict(PASS=True,source=ref.sha(__file__),reference=ref.sha(ROOT/'shakhov_reference.py'),manifest=ref.sha(ROOT/'manifest.json'),collision_invariants= float(inv),stress_BE_error=float(stress),heat_BE_error=float(heat),scalar_BE_error=float(fourth),equilibrium_error=float(eqerr),independent_transverse_error=float(transverse),viscosity_match_error=float(viscosity))
    result['all_case_collision_invariants']=case_errors;result['restart_bitwise']=True
    ref.dump(ROOT/'PASS.json',result);print(json.dumps(result))
def budgets(f,x,v,w,c,out):
    a=ref.moments(f,v,w);rho,u,T,q,sig,R,D,S=a.T;cx=v[None,:]-u[:,None];g,h,k=f
    F=.5*np.sum(v*w*(cx**3*g+cx*h),axis=1)
    J=np.sum(v*w*(cx**4*g+2*cx**2*h+k),axis=1)
    K=2*F;L=np.sum(v*w*(1.5*cx**2*g+.5*h),axis=1)
    alpha=1-1/c['Pr'];tau=(1-alpha)*c['mu1']*T**(c['omega']-1)/rho
    E=target(v,rho,u,T,sig,alpha);eg,eh,ek=E
    # Actual finite-grid target moments about the current gas mean.
    qG=.5*np.sum(w*(cx**3*eg+cx*eh),axis=1)
    dG=np.sum(w*(cx**4*eg+2*cx**2*eh+ek),axis=1)-15*rho*T*T
    Qq=(qG-q)/tau;Qd=(dG-D)/tau;du=np.gradient(u,x,edge_order=2)
    integrand=Qq-du*L
    Fbud=F[0]+np.r_[0,np.cumsum(.5*(integrand[1:]+integrand[:-1])*np.diff(x))]
    Sbud=2*(Fbud-u*q)-5*rho*T*T-7*T*sig
    Db=dG-tau*(np.gradient(J,x,edge_order=2)+4*du*K);Rb=Sbud-Db/3
    heatres=np.gradient(F,x,edge_order=2)+du*L-Qq
    dres=np.gradient(J,x,edge_order=2)+4*du*K-Qd
    mask=abs(S)>.02*np.max(abs(S))
    def err(rebuilt,direct):return float(np.linalg.norm((rebuilt-direct)[mask])/max(np.linalg.norm(direct[mask]),1e-30))
    report=dict(active_mask='abs(S_direct)>0.02*max(abs(S_direct)); no shock re-alignment',S_relative_L2=err(Sbud,S),Delta_relative_L2=err(Db,D),R_relative_L2=err(Rb,R),heat_differential_relative_L2=float(np.linalg.norm(heatres)/max(np.linalg.norm(Qq)+np.linalg.norm(du*L),1e-30)),scalar_differential_relative_L2=float(np.linalg.norm(dres)/max(np.linalg.norm(Qd)+np.linalg.norm(4*du*K),1e-30)),finite_target_q_max=float(np.max(abs(qG))),finite_target_Delta_error=float(np.max(abs(dG-3*alpha**2*sig**2/rho))),note='Steady-form reconstruction: transient and discretization residuals remain; not a stationarity certificate.')
    ref.save(out/'budgets.npz',x=x,direct=a,S_reconstructed=Sbud,Delta_reconstructed=Db,R_reconstructed=Rb,F=F,J=J,K=K,L=L,Qq=Qq,QDelta=Qd,heat_residual=heatres,scalar_residual=dres,mask=mask)
    ref.dump(out/'BUDGETS.json',report);return report
def run(c):
    out=ROOT/'runs'/c['name'];out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256((json.dumps(c,sort_keys=True)+ref.sha(__file__)+ref.sha(ROOT/'shakhov_reference.py')).encode()).hexdigest()
    x,v,w,f,left,right=ref.setup(c);dt=c['CFL']*2*c['L']/c['nx']/c['vmax'];step=0;stable=0;prev=None;drift=None;rhs=None
    cp=out/'checkpoint.npz'
    if cp.exists():
        z=np.load(cp);assert str(z['signature'])==signature;f=z['f'];step=int(z['step']);stable=int(z['stable']);prev=z['prev'];prev=None if prev.size==0 else prev
    start=time.monotonic();every=max(1,round(2/dt));nmax=int(np.ceil(c['tmax']/dt));reason='time_limit'
    ref.dump(out/'START.json',dict(pid=os.getpid(),config=c,signature=signature,dt=dt))
    while step<nmax and stable<5 and not STOP:
        fnew=advance(f,v,w,left,right,c,dt);step+=1
        if step%every==0:
            a=ref.moments(fnew,v,w);before=ref.moments(f,v,w)
            scale=np.maximum(np.max(abs(a),axis=0),1e-8)
            rhs=float(np.max(np.max(abs(a-before),axis=0)/(dt*scale)))
            if prev is not None:drift=float(np.max(np.max(abs(a-prev),axis=0)/scale))
            stable=stable+1 if drift is not None and drift<1e-7 and rhs<1e-7 else 0;prev=a.copy()
            ref.dump(out/'PROGRESS.json',dict(step=step,t=step*dt,rhs=rhs,drift=drift,stable=stable,seconds=time.monotonic()-start))
            print(json.dumps(dict(step=step,t=step*dt,rhs=rhs,drift=drift,stable=stable,seconds=time.monotonic()-start)),flush=True)
        f=fnew
        if step%1000==0 or STOP:ref.save(cp,f=f,step=step,stable=stable,prev=np.array([]) if prev is None else prev,signature=signature)
        if time.monotonic()-start>c['wall_budget']:reason='cooperative_wall_budget';break
    if stable>=5:reason='stationary_checks_passed'
    elif step>=nmax:reason='physical_time_limit'
    elif STOP:reason='signal_checkpoint'
    ref.save(cp,f=f,step=step,stable=stable,prev=np.array([]) if prev is None else prev,signature=signature)
    a=ref.moments(f,v,w);after=ref.moments(advance(f,v,w,left,right,c,dt),v,w)
    final_rhs=float(np.max(np.max(abs(after-a),axis=0)/(dt*np.maximum(np.max(abs(a),axis=0),1e-8))))
    face=np.where((v>=0)[None,None,:],np.concatenate([left[:,None,:],f],axis=1),np.concatenate([f,right[:,None,:]],axis=1))
    flux=np.stack([np.sum(face[0]*v*w,axis=1),np.sum(face[0]*v*v*w,axis=1),.5*np.sum((v*v*face[0]+face[1])*v*w,axis=1)],axis=1)
    ref.save(out/'profiles.npz',x=x,fields=a,columns=np.array(['rho','u','T','q','sigma','R','Delta','S']))
    b=budgets(f,x,v,w,c,out)
    status=dict(config=c,signature=signature,step=step,time=step*dt,dt=dt,reason=reason,stationary=stable>=5,stable=stable,final_rhs=final_rhs,last_drift=drift,seconds=time.monotonic()-start,min_reduced=float(np.min(f)),min_rho=float(np.min(a[:,0])),min_T=float(np.min(a[:,2])),face_flux_relative_spans=(np.ptp(flux,axis=0)/np.maximum(abs(flux[0]),1e-12)).tolist(),budgets=b,sha256={p.name:ref.sha(p) for p in [cp,out/'profiles.npz',out/'budgets.npz']})
    ref.dump(out/'STATUS.json',status);print(json.dumps(status),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gate',action='store_true');ap.add_argument('--index',type=int,default=0);args=ap.parse_args()
    if args.gate:gate()
    else:
        p=json.loads((ROOT/'PASS.json').read_text());assert p['PASS'] and p['source']==ref.sha(__file__) and p['reference']==ref.sha(ROOT/'shakhov_reference.py') and p['manifest']==ref.sha(ROOT/'manifest.json')
        run(json.loads((ROOT/'manifest.json').read_text())['cases'][args.index])
