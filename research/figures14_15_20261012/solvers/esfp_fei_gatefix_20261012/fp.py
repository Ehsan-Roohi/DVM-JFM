"""Deterministic reduced ES-FP shock. Fei Eq25 adaptive beta; no tensor clipping.
g,h,k = transverse integrals weighted by 1, r_perp^2, r_perp^4.
FP sources: Lx g; Lx h+(-2h+4Pi_y*g)/tau;
Lx k+(-4k+16Pi_y*h)/tau. tau=2*(1-beta)/rho for mu=T; 4.5/rho on the fixed branch.
Implicit Chang-Cooper with explicitly measured moment-fitting corrections.
"""
import argparse,ctypes,hashlib,json,os,signal,time
from pathlib import Path
import numpy as np
import shakhov_reference as ref
ROOT=Path(__file__).resolve().parent
LIB=ctypes.CDLL(str(ROOT/'fp_kernel.so'))
PTR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
LIB.collide.argtypes=[ctypes.c_int,ctypes.c_int,PTR,ctypes.c_double,ctypes.c_double,PTR,PTR,PTR]
LIB.collide.restype=ctypes.c_int
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
def setup(c):
    dv=2*c['vmax']/c['nv'];v=-c['vmax']+(np.arange(c['nv'])+.5)*dv;w=np.full(c['nv'],dv)
    m=c['mach'];u1=m*np.sqrt(5/3);rho2=4*m*m/(m*m+3);T2=(5*m*m-1)/(4*rho2)
    left=ref.eq(v,np.array([1.]),np.array([u1]),np.array([1.]))[:,0]
    right=ref.eq(v,np.array([rho2]),np.array([u1/rho2]),np.array([T2]))[:,0]
    x=-c['L']+(np.arange(c['nx'])+.5)*(2*c['L']/c['nx']);mix=(1+np.tanh(x/3))/2
    f=np.ascontiguousarray(left[:,None,:]*(1-mix[None,:,None])+right[:,None,:]*mix[None,:,None])
    return x,v,w,f,left,right
def collision(f,v,w,dt):
    f=np.ascontiguousarray(f);out=np.empty_like(f);diag=np.zeros(5)
    code=LIB.collide(f.shape[1],len(v),v,float(w[0]),dt,f,out,diag)
    if code:raise FloatingPointError(f'ESFP collision guard code {code}; no clipping or retry')
    return out,diag
def advance(f,v,w,left,right,c,dt):
    face=np.where((v>=0)[None,None,:],np.concatenate([left[:,None,:],f],axis=1),np.concatenate([f,right[:,None,:]],axis=1))
    transported=f-dt/(2*c['L']/c['nx'])*v[None,None,:]*(face[:,1:]-face[:,:-1])
    if not np.isfinite(transported).all() or np.min(transported)<0:raise FloatingPointError('Transport positivity')
    new,diag=collision(transported,v,w,dt)
    # Exact discrete finite-volume balance including reservoir fluxes.
    def inv(a):return np.stack([np.sum(a[0]*w,axis=1),np.sum(a[0]*v*w,axis=1),.5*np.sum((a[0]*v*v+a[1])*w,axis=1)],axis=1)
    flux=np.stack([np.sum(face[0]*v*w,axis=1),np.sum(face[0]*v*v*w,axis=1),.5*np.sum((face[0]*v*v+face[1])*v*w,axis=1)],axis=1)
    before=np.sum(inv(f),axis=0);after=np.sum(inv(new),axis=0)
    bal=after-before+dt/(2*c['L']/c['nx'])*(flux[-1]-flux[0])
    defect=float(np.max(abs(bal)/np.maximum(abs(before),1)))
    if defect>3e-11:raise FloatingPointError(f'Global finite-volume conservation {defect}')
    return new,diag,defect
def gate():
    tests=[]
    for nv in [400,800,1600]:
        c=dict(mach=2,nx=12,L=30,nv=nv,vmax=40)
        x,v,w,f,left,right=setup(c);a=ref.moments(f,v,w);dt=.01
        new,diag=collision(f,v,w,dt);b=ref.moments(new,v,w);r=dt*a[:,0]/4.5
        expected_q=a[:,3]/(1+3*r)
        expected_D=(a[:,6]-15*r*b[:,4]**2/b[:,0])/(1+4*r)
        errq=float(np.max(abs(b[:,3]-expected_q))/max(np.max(abs(a[:,3])),1e-12))
        errD=float(np.max(abs(b[:,6]-expected_D))/max(np.max(abs(a[:,6])),1e-12))
        stress=float(np.max(abs(b[:,4]-a[:,4]/(1+4.5*r))))
        invariants=float(np.max(abs(b[:,:3]-a[:,:3])/np.maximum(abs(a[:,:3]),1)))
        realizability=float(np.min(new[0]*new[2]-new[1]**2))
        eq=ref.eq(v,np.ones(12),np.ones(12)*2,np.ones(12))
        equilibrated,_=collision(eq,v,w,dt)
        eqerr=float(np.max(abs(equilibrated-eq)))
        assert invariants<1e-11 and stress<1e-10 and eqerr<1e-10 and realizability>=-1e-14
        assert errq<.002 and errD<.002
        tests.append(dict(nv=nv,invariants=invariants,stress_BE_error=stress,heat_BE_discretization_error=errq,scalar_BE_discretization_error=errD,equilibrium_error=eqerr,diagnostics=diag.tolist(),minimum_realizability=realizability))
    assert tests[-1]['heat_BE_discretization_error']<tests[0]['heat_BE_discretization_error']/8
    assert tests[-1]['scalar_BE_discretization_error']<tests[0]['scalar_BE_discretization_error']/8
    c=dict(mach=2,nx=12,L=30,nv=800,vmax=40);x,v,w,f,l,r=setup(c)
    first,_,_=advance(f,v,w,l,r,c,.001);ref.save(ROOT/'gate_restart.npz',f=first)
    a,_,_=advance(first,v,w,l,r,c,.001);b,_,_=advance(np.load(ROOT/'gate_restart.npz')['f'],v,w,l,r,c,.001)
    assert np.array_equal(a,b)
    result=dict(PASS=True,tests=tests,restart_bitwise=True,binding={n:ref.sha(ROOT/n) for n in ['esfp.py','kernel.c','kernel.so','shakhov_reference.py','manifest.json']},scope='Pure collision/transport fixtures only; no shock stationarity or spatial qualification')
    ref.dump(ROOT/'PASS.json',result);print(json.dumps(result))
def budgets(f,x,v,w,c,out):
    a=ref.moments(f,v,w);rho,u,T,q,sig,R,D,S=a.T;cx=v[None,:]-u[:,None];g,h,k=f;anis=np.maximum(sig/rho,-sig/(2*rho));beta=np.full(len(rho),-1.25);maskbeta=anis>0;beta[maskbeta]=np.maximum(beta[maskbeta],-(1-1e-12)*T[maskbeta]/anis[maskbeta]);tau=2*(1-beta)/rho
    F=.5*np.sum(v*w*(cx**3*g+cx*h),axis=1);J=np.sum(v*w*(cx**4*g+2*cx**2*h+k),axis=1)
    L=np.sum(v*w*(1.5*cx**2*g+.5*h),axis=1);K=2*F
    Qq=-3*q/tau;Qd=(-4*D+12*beta*sig**2/rho)/tau;du=np.gradient(u,x,edge_order=2)
    rhs=Qq-du*L;Fb=F[0]+np.r_[0,np.cumsum(.5*(rhs[1:]+rhs[:-1])*np.diff(x))]
    Sb=2*(Fb-u*q)-5*rho*T*T-7*T*sig
    Db=-(tau/4)*(np.gradient(J,x,edge_order=2)+4*du*K)+3*beta*sig**2/rho;Rb=Sb-Db/3
    mask=abs(S)>.02*np.max(abs(S));error=lambda pred,true:float(np.linalg.norm((pred-true)[mask])/max(np.linalg.norm(true[mask]),1e-30))
    report=dict(S_relative_L2=error(Sb,S),Delta_relative_L2=error(Db,D),R_relative_L2=error(Rb,R),note='Steady-form continuum reconstruction; temporal and discrete-velocity defects remain. Not stationarity proof.')
    ref.save(out/'budgets.npz',x=x,direct=a,S_reconstructed=Sb,Delta_reconstructed=Db,R_reconstructed=Rb,Qq=Qq,QDelta=Qd,mask=mask)
    ref.dump(out/'BUDGETS.json',report);return report
def run(index):
    gate=json.loads((ROOT/'PASS.json').read_text());assert gate['PASS']
    for n,s in gate['binding'].items():assert ref.sha(ROOT/n)==s,n
    c=json.loads((ROOT/'manifest.json').read_text())['cases'][index]
    signature=hashlib.sha256(json.dumps(dict(config=c,binding=gate['binding']),sort_keys=True).encode()).hexdigest()
    out=ROOT/'runs'/c['name'];out.mkdir(parents=True,exist_ok=True);cp=out/'checkpoint.npz'
    x,v,w,f,left,right=setup(c);dt=c['CFL']*2*c['L']/c['nx']/c['vmax'];step=stable=0;prev=None
    if cp.exists():
        z=np.load(cp,allow_pickle=False);assert str(z['signature'])==signature
        f=z['f'];step=int(z['step']);stable=int(z['stable']);prev=z['prev'];prev=None if not prev.size else prev
    start=time.monotonic();every=max(1,round(2/dt));nmax=int(np.ceil(c['tmax']/dt));drift=None;maxdiag=np.zeros(5);minPi=float('inf');maxbalance=0
    reason='physical_time_limit';ref.dump(out/'START.json',dict(pid=os.getpid(),signature=signature,config=c,dt=dt))
    try:
        while step<nmax and stable<5 and not STOP:
            fn,d,balance=advance(f,v,w,left,right,c,dt)
            maxdiag=np.maximum(maxdiag,d);minPi=min(minPi,d[3]);maxbalance=max(maxbalance,balance)
            if (step+1)%every==0:
                a=ref.moments(fn,v,w);old=ref.moments(f,v,w);scale=np.maximum(np.max(abs(a),axis=0),1e-8)
                rhs=float(np.max(abs(a-old)/(dt*scale)));drift=None if prev is None else float(np.max(abs(a-prev)/scale))
                stable=stable+1 if drift is not None and drift<1e-7 and rhs<1e-7 else 0;prev=a.copy()
                rec=dict(step=step+1,t=(step+1)*dt,rhs=rhs,drift=drift,stable=stable,seconds=time.monotonic()-start)
                ref.dump(out/'PROGRESS.json',rec);print(json.dumps(rec),flush=True)
            f=fn;step+=1
            if step%1000==0 or STOP:ref.save(cp,f=f,step=step,stable=stable,prev=np.array([]) if prev is None else prev,signature=signature)
            if time.monotonic()-start>c['wall_budget']:reason='cooperative_wall_budget';break
        if STOP:reason='signal_checkpoint'
        if stable>=5:reason='stationary_checks_passed'
        ref.save(cp,f=f,step=step,stable=stable,prev=np.array([]) if prev is None else prev,signature=signature)
        a=ref.moments(f,v,w);extra,_,_=advance(f,v,w,left,right,c,dt);nexta=ref.moments(extra,v,w)
        rhs=float(np.max(abs(nexta-a)/(dt*np.maximum(np.max(abs(a),axis=0),1e-8))))
        ref.save(out/'profiles.npz',x=x,fields=a,columns=np.array(['rho','u','T','q','sigma','R','Delta','S']))
        report=dict(config=c,signature=signature,step=step,time=step*dt,dt=dt,reason=reason,stationary=stable>=5 and rhs<1e-7,final_rhs=rhs,last_drift=drift,seconds=time.monotonic()-start,max_conservation_defect=maxbalance,min_diffusion_eigenvalue=minPi,max_fit_error=float(maxdiag[0]),max_drift_shift=float(maxdiag[1]),max_relative_diffusion_fit=float(maxdiag[2]),max_collision_invariant_error=float(maxdiag[4]),min_reduced=float(np.min(f)),min_rho=float(np.min(a[:,0])),min_T=float(np.min(a[:,2])),budgets=budgets(f,x,v,w,c,out),sha256={p.name:ref.sha(p) for p in [cp,out/'profiles.npz',out/'budgets.npz']})
        ref.dump(out/'STATUS.json',report);print(json.dumps(report),flush=True)
    except BaseException as e:
        ref.save(cp,f=f,step=step,stable=stable,prev=np.array([]) if prev is None else prev,signature=signature)
        ref.dump(out/'FAILED.json',dict(error=repr(e),last_completed_step=step,checkpoint_sha256=ref.sha(cp)));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gate',action='store_true');p.add_argument('--index',type=int,default=0);a=p.parse_args()
    gate() if a.gate else run(a.index)
