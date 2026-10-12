"""CPU Shakhov DVM matching R26 transport parameters, NOT its Boltzmann operator.

Exact transverse integrations g=integral f, h=integral r_perp^2 f,
k=integral r_perp^4 f retain all axial fourth moments. No clipping.
"""
import os,sys,json,time,hashlib,signal,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
STOP=False
def stop(*args):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop)
if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,stop)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,a):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(a,indent=2,allow_nan=False));os.replace(tmp,p)
def eq(v,rho,u,T):
    G=rho[:,None]/np.sqrt(2*np.pi*T[:,None])*np.exp(-(v[None,:]-u[:,None])**2/(2*T[:,None]))
    return np.stack([G,2*T[:,None]*G,8*T[:,None]**2*G])
def setup(c):
    v=np.linspace(-c['vmax'],c['vmax'],c['nv']);w=np.full_like(v,v[1]-v[0]);w[[0,-1]]*=.5
    dx=2*c['L']/c['nx'];x=-c['L']+(np.arange(c['nx'])+.5)*dx
    m=c['mach'];ga=5/3;u1=m*np.sqrt(ga);rho2=4*m*m/(m*m+3);T2=(5*m*m-1)/4/rho2
    left=eq(v,np.array([1.]),np.array([u1]),np.array([1.]))[:,0]
    right=eq(v,np.array([rho2]),np.array([u1/rho2]),np.array([T2]))[:,0]
    H=(1+np.tanh(x/3))/2
    f=left[:,None,:]*(1-H[None,:,None])+right[:,None,:]*H[None,:,None]
    return x,v,w,f,left,right
def moments(f,v,w):
    g,h,k=f
    rho=np.sum(g*w,axis=1);u=np.sum(g*w*v,axis=1)/rho
    T=(np.sum((g*v*v+h)*w,axis=1)/rho-u*u)/3
    if not (np.isfinite(f).all() and (rho>0).all() and (T>0).all()): raise RuntimeError('Nonfinite or nonpositive macrostate; no clipping')
    cx=v[None,:]-u[:,None];cx2=cx*cx
    q=.5*np.sum((cx*cx2*g+cx*h)*w,axis=1)
    sig=np.sum(cx2*g*w,axis=1)-rho*T
    fourth=np.sum((cx2**2*g+2*cx2*h+k)*w,axis=1)
    D=fourth-15*rho*T*T
    R=np.sum((cx2**2*g+cx2*h)*w,axis=1)-fourth/3-7*T*sig
    return np.column_stack([rho,u,T,q,sig,R,D,R+D/3])
def collide(f,v,w,c,dt):
    a=moments(f,v,w);rho,u,T,q=a[:,:4].T
    E=eq(v,rho,u,T);cx=v[None,:]-u[:,None]
    # mu1=rho1=T1=1 and omega=1: tau=mu/(rho*T)=1/rho.
    tau=c['mu1']*T**(c['omega']-1)/rho
    decay=np.exp(-dt/tau)[None,:,None]
    B=(np.exp(-c['Pr']*dt/tau)-np.exp(-dt/tau))[None,:,None]
    A=q[:,None]*cx/(5*rho[:,None]*T[:,None]**2)
    polys=cx[None,:,:]**2/T[None,:,None]+np.array([-3.,-1.,1.])[:,None,None]
    return E+(f-E)*decay+E*A[None,:,:]*polys*B
def advance(f,v,w,left,right,c,dt):
    dx=2*c['L']/c['nx'];pos=v>=0
    diff=np.empty_like(f)
    diff[:,:,pos]=f[:,:,pos]-np.concatenate([left[:,None,pos],f[:,:-1,pos]],axis=1)
    diff[:,:,~pos]=np.concatenate([f[:,1:,~pos],right[:,None,~pos]],axis=1)-f[:,:,~pos]
    return collide(f-dt/dx*v[None,None,:]*diff,v,w,c,dt)
def save(path,**a):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as h: np.savez(h,**a)
    os.replace(tmp,path)
def gate(c):
    x,v,w,f,left,right=setup(dict(c,nx=12))
    a=moments(f,v,w);dt=.01;out=collide(f,v,w,c,dt);b=moments(out,v,w)
    E=np.exp(-dt*a[:,0]);Q=np.exp(-c['Pr']*dt*a[:,0])
    errors={'invariants':float(np.max(abs(b[:,:3]-a[:,:3]))),
            'q_decay':float(np.max(abs(b[:,3]-Q*a[:,3]))),
            'stress_decay':float(np.max(abs(b[:,4]-E*a[:,4]))),
            'R_decay':float(np.max(abs(b[:,5]-E*a[:,5]))),
            'Delta_decay':float(np.max(abs(b[:,6]-E*a[:,6])))}
    assert max(errors.values())<1e-8,errors
    cb=dict(c,Pr=1.);rr=collide(f,v,w,cb,dt);em=eq(v,a[:,0],a[:,1],a[:,2])
    assert np.max(abs(rr-(em+(f-em)*E[None,:,None])))<1e-12
    # Independent 3-D transverse quadrature of Shakhov's target, including k.
    vt=np.linspace(-12,12,121);wt=np.full_like(vt,.2);wt[[0,-1]]*=.5
    rad=vt[:,None]**2+vt[None,:]**2;ww=wt[:,None]*wt[None,:]
    rho,u,T,q=1.2,2.,1.7,.2
    G=eq(v,np.array([rho]),np.array([u]),np.array([T]))[:,0]
    cx=v-u;A=(1-c['Pr'])*q*cx/(5*rho*T*T)
    reduced=G*(1+A[None,:]*(cx[None,:]**2/T+np.array([-3.,-1.,1.])[:,None]))
    numeric=[]
    for j in range(3):
        F=(rho/(2*np.pi*T)**1.5*np.exp(-(cx[:,None,None]**2+rad)/(2*T))
           *(1+A[:,None,None]*((cx[:,None,None]**2+rad)/T-5)))
        numeric.append(np.sum(F*(rad**j*ww)[None,:,:],axis=(1,2)))
    errors['transverse_3d_quadrature']=float(np.max(abs(np.array(numeric)-reduced)))
    assert errors['transverse_3d_quadrature']<1e-10,errors
    uniform=np.repeat(left[:,None,:],12,axis=1)
    assert np.max(abs(advance(uniform,v,w,left,left,dict(c,nx=12),dt)-uniform))<1e-12
    cc=dict(c,nx=12);direct=f.copy()
    for _ in range(4):direct=advance(direct,v,w,left,right,cc,dt)
    split=f.copy()
    for _ in range(2):split=advance(split,v,w,left,right,cc,dt)
    td=ROOT/'gate_restart';td.mkdir(exist_ok=True)
    cp=td/'restart.npz';save(cp,f=split)
    split=np.load(cp,allow_pickle=False)['f']
    for _ in range(2):split=advance(split,v,w,left,right,cc,dt)
    assert np.array_equal(split,direct)
    return dict(PASS=True,errors=errors,restart_bitwise=True,source=sha(__file__))
def run(c,out):
    out.mkdir(parents=True,exist_ok=True)
    sig=hashlib.sha256((sha(__file__)+json.dumps(c,sort_keys=True)).encode()).hexdigest()
    if (out/'DONE.json').exists():
        assert json.loads((out/'DONE.json').read_text())['signature']==sig
        print('Already finished');return
    x,v,w,f,left,right=setup(c);dt=c['CFL']*2*c['L']/c['nx']/c['vmax']
    step=0;stable=0;prev=None;drift=1.
    cp=out/'checkpoint.npz'
    if cp.exists():
        ck=np.load(cp,allow_pickle=False);assert str(ck['signature'])==sig
        f=ck['f'];step=int(ck['step']);stable=int(ck['stable']);prev=ck['prev'] if ck['prev'].size else None
    start=time.time();maxsteps=int(np.ceil(c['tmax']/dt));interval=max(1,int(round(2/dt)))
    while step<maxsteps and stable<5 and not STOP:
        f=advance(f,v,w,left,right,c,dt);step+=1
        if step%interval==0:
            a=moments(f,v,w)
            if prev is not None:
                drift=float(np.max(np.max(abs(a-prev),axis=0)/np.maximum(np.max(abs(a),axis=0),1e-8)))
                stable=stable+1 if drift<1e-7 else 0
            prev=a.copy()
            print(json.dumps(dict(step=step,t=step*dt,drift=drift,stable=stable,seconds=time.time()-start)),flush=True)
        if step%1000==0 or STOP:
            save(cp,f=f,step=step,stable=stable,prev=np.array([]) if prev is None else prev,signature=sig)
        if time.time()-start>6600:break
    save(cp,f=f,step=step,stable=stable,prev=np.array([]) if prev is None else prev,signature=sig)
    a=moments(f,v,w)
    negative={name:float(np.max(np.sum(np.maximum(-f[i],0)*w,axis=1)/np.maximum(np.sum(abs(f[i])*w,axis=1),1e-300))) for i,name in enumerate(['g','h','k'])}
    face=np.where((v>=0)[None,None,:],np.concatenate([left[:,None,:],f],axis=1),np.concatenate([f,right[:,None,:]],axis=1))
    flux=np.column_stack([np.sum(face[0]*v*w,axis=1),np.sum(face[0]*v*v*w,axis=1),.5*np.sum((v*v*face[0]+face[1])*v*w,axis=1)])
    status=dict(signature=sig,step=step,time=step*dt,dt=dt,stationary=stable>=5,stable_count=stable,last_drift=drift,
                negative_fractions=negative,config=c,source=sha(__file__),collision_model='Shakhov; not Maxwell Boltzmann',
                face_flux_relative_spans=(np.ptp(flux,axis=0)/np.maximum(abs(flux[0]),1e-12)).tolist(),
                max_rss_note='Measure using /usr/bin/time or Slurm sacct',finished=step>=maxsteps or stable>=5)
    save(out/'profiles.npz',x=x,fields=a,columns=np.array(['rho','u','T','q','sigma','R','Delta','S']))
    dump(out/'STATUS.json',status)
    if status['finished']:
        status['sha256']={p.name:sha(p) for p in [cp,out/'profiles.npz']}
        dump(out/'DONE.json',status)
    print(json.dumps(status),flush=True)
    return status
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gate',action='store_true');ap.add_argument('--index',type=int,default=0);ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    manifest=json.loads((ROOT/'manifest.json').read_text());c=manifest['cases'][args.index]
    if args.gate:
        result=gate(c);result['manifest']=sha(ROOT/'manifest.json');dump(ROOT/'PASS.json',result);print(json.dumps(result))
    elif args.benchmark:
        x,v,w,f,l,r=setup(c);start=time.time()
        for _ in range(100):f=advance(f,v,w,l,r,c,.8*2*c['L']/c['nx']/c['vmax'])
        print(json.dumps(dict(seconds_100_steps=time.time()-start,distribution_MiB=f.nbytes/2**20)))
    else:
        passed=json.loads((ROOT/'PASS.json').read_text());assert passed['PASS'] and passed['source']==sha(__file__) and passed['manifest']==sha(ROOT/'manifest.json')
        run(c,ROOT/'runs'/c['name'])
