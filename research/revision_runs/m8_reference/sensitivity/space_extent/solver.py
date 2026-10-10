import os, sys, json, math, time, hashlib, signal
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
STOP=False
def stop(*_):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGUSR1,stop)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup(c,device):
    v=torch.linspace(c['vmin'],c['vmax'],c['nv'],device=device,dtype=torch.float64)
    w=torch.ones_like(v)*(v[1]-v[0]);w[0]*=.5;w[-1]*=.5
    x=c['xmin']+(torch.arange(c['nx'],device=device,dtype=torch.float64)+.5)*(c['xmax']-c['xmin'])/c['nx']
    ga=c['gamma'];m=c['M'];r=(ga+1)*m*m/((ga-1)*m*m+2)
    u1=m*math.sqrt(ga);T2=(1+2*ga/(ga+1)*(m*m-1))/r
    def eq(rho,u,T):
        G=(rho/torch.sqrt(2*math.pi*T))[:,None]*torch.exp(-(v[None,:]-u[:,None])**2/(2*T[:,None]))
        return G,2*T[:,None]*G
    def boundary(rho,u,T):
        vals=[torch.tensor([z],device=device,dtype=torch.float64) for z in (rho,u,T)]
        return tuple(z[0] for z in eq(*vals))
    left=boundary(1.,u1,1.);right=boundary(r,u1/r,T2)
    H=.5*(1+torch.tanh(x/3))
    # Fresh positive bimodal distribution, not a BGK checkpoint.
    g=(1-H[:,None])*left[0]+H[:,None]*right[0]
    h=(1-H[:,None])*left[1]+H[:,None]*right[1]
    return x,v,w,g,h,left,right,eq
def macro(g,h,v,w):
    rho=(g*w).sum(1);u=(g*w*v).sum(1)/rho
    T=((g*w*v*v+h*w).sum(1)/rho-u*u)/3
    if not bool(torch.isfinite(T).all() and (T>0).all() and (rho>0).all()): raise RuntimeError('Invalid macroscopic state; no clipping allowed')
    cx=v[None,:]-u[:,None]
    q=.5*((cx**3*g+cx*h)*w).sum(1)
    return rho,u,T,q
def collide(g,h,v,w,eq,c,dt):
    rho,u,T,q=macro(g,h,v,w);G,H=eq(rho,u,T)
    cx=v[None,:]-u[:,None]
    # Eq35, rho1=T1=R=lambda1=1. tau=mu/p, mu=mu_ref*T**omega.
    mu0=30*math.sqrt(2*math.pi)/(4*(7-2*c['omega'])*(5-2*c['omega']))
    tau=mu0*T**(c['omega']-1)/rho
    a=dt/tau;E=torch.exp(-a);B=torch.exp(-c['Pr']*a)-E
    A=q[:,None]*cx/(5*rho[:,None]*T[:,None]**2)
    # Analytic integration of the decaying q in the Shakhov target.
    gn=G+(g-G)*E[:,None]+G*A*(cx*cx/T[:,None]-3)*B[:,None]
    hn=H+(h-H)*E[:,None]+H*A*(cx*cx/T[:,None]-1)*B[:,None]
    return gn,hn
def advance(g,h,v,w,eq,left,right,c,dt):
    dx=(c['xmax']-c['xmin'])/c['nx']
    out=[]
    for f,l,r in [(g,left[0],right[0]),(h,left[1],right[1])]:
        fl=torch.cat([l[None,:],f[:-1]]);fr=torch.cat([f[1:],r[None,:]])
        out.append(f-dt/dx*v[None,:]*torch.where(v[None,:]>=0,f-fl,fr-f))
    return collide(*out,v,w,eq,c,dt)
def fields(g,h,v,w):
    rho,u,T,q=macro(g,h,v,w)
    sig=((v[None,:]-u[:,None])**2*g*w).sum(1)-rho*T
    return torch.stack([rho,u,T,q,sig],dim=1)
def checkpoint(path,g,h,step,signature,prev,stable):
    assert bool(torch.isfinite(g).all() and torch.isfinite(h).all())
    tmp=path.with_suffix('.tmp')
    torch.save({'g':g.cpu(),'h':h.cpu(),'step':step,'signature':signature,'previous':prev,'stable':stable},tmp);os.replace(tmp,path)
def gate(c,device):
    x,v,w,g,h,l,r,eq=setup(c,device)
    # Bimodal cells exercise nonzero heat flux; collision moment laws independent of target implementation.
    a,b=g[::max(1,c['nx']//7)],h[::max(1,c['nx']//7)]
    rho,u,T,q=macro(a,b,v,w);dt=.01
    aa,bb=collide(a,b,v,w,eq,c,dt);rr,uu,tt,qq=macro(aa,bb,v,w)
    mu0=30*math.sqrt(2*math.pi)/(4*(7-2*c['omega'])*(5-2*c['omega']))
    tau=mu0*T**(c['omega']-1)/rho
    errors={'rho':float(((rr-rho).abs()/rho).max()),'u':float(((uu-u).abs()/(1+u.abs())).max()),'T':float(((tt-T).abs()/T).max()),'heat_decay':float((qq-q*torch.exp(-c['Pr']*dt/tau)).abs().max()/(1+q.abs().max()))}
    assert max(errors.values())<1e-8,errors
    # Pr=1 must reduce to a Maxwellian BGK collision.
    cb=dict(c,Pr=1.);ag,ah=collide(a,b,v,w,eq,cb,dt);G,H=eq(rho,u,T);E=torch.exp(-dt/tau)[:,None]
    assert torch.allclose(ag,G+(a-G)*E,atol=1e-13,rtol=1e-13)
    gs,hs=g,h;d=c['CFL']*(c['xmax']-c['xmin'])/c['nx']/max(abs(c['vmin']),abs(c['vmax']))
    for _ in range(4): gs,hs=advance(gs,hs,v,w,eq,l,r,c,d)
    gr,hr=g,h
    for _ in range(2): gr,hr=advance(gr,hr,v,w,eq,l,r,c,d)
    checkpoint(ROOT/'gate_restart.pt',gr,hr,2,'gate',None,0)
    ck=torch.load(ROOT/'gate_restart.pt',weights_only=False,map_location=device)
    gr,hr=ck['g'],ck['h']
    for _ in range(2): gr,hr=advance(gr,hr,v,w,eq,l,r,c,d)
    assert torch.equal(gs,gr) and torch.equal(hs,hr)
    if device=='cuda':
        cpu=setup(c,'cpu');cg,ch=advance(cpu[3],cpu[4],cpu[1],cpu[2],cpu[7],cpu[5],cpu[6],c,d)
        gg,gh=advance(g,h,v,w,eq,l,r,c,d)
        assert torch.allclose(cg,gg.cpu(),rtol=1e-10,atol=1e-12) and torch.allclose(ch,gh.cpu(),rtol=1e-10,atol=1e-12)
    result={'PASS':True,'errors':errors,'restart_bitwise':True,'source':sha(__file__),'config':sha(ROOT/'config.json'),'device':device,'mu_ref':mu0,'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}
    (ROOT/('GPU_PASS.json' if device=='cuda' else 'CPU_PASS.json')).write_text(json.dumps(result,indent=2));print(result,flush=True)
def run(c,device):
    passed=json.loads((ROOT/'GPU_PASS.json').read_text());assert passed['PASS'] and passed['source']==sha(__file__) and passed['config']==sha(ROOT/'config.json')
    signature=sha(__file__)+sha(ROOT/'config.json');path=ROOT/'checkpoint.pt'
    x,v,w,g,h,l,r,eq=setup(c,device);step=0;prev=None;stable=0
    if path.exists():
        ck=torch.load(path,weights_only=False,map_location='cpu');assert ck['signature']==signature
        g=ck['g'].to(device);h=ck['h'].to(device);step=ck['step'];prev=ck['previous'];stable=ck['stable'];fields(g,h,v,w)
    dt=c['CFL']*(c['xmax']-c['xmin'])/c['nx']/max(abs(c['vmin']),abs(c['vmax']));end=math.ceil(c['max_time']/dt);started=time.monotonic()
    print('START',step,'dt',dt,'end',end,flush=True)
    with torch.no_grad():
        while step<end and stable<c['stationarity_checks']:
            g,h=advance(g,h,v,w,eq,l,r,c,dt);step+=1
            if step%c['snapshot_every']==0:
                y=fields(g,h,v,w).cpu().numpy();err=None
                if prev is not None:
                    err=float(np.max(np.max(abs(y-prev),axis=0)/np.maximum(np.max(abs(y),axis=0),1.)))
                    stable=stable+1 if err<c['stationarity_relative_linf'] else 0
                prev=y
                np.savez(ROOT/f'profile_{step:08d}.npz',x=x.cpu().numpy(),fields=y,columns=['rho','u','T','q','sig'],step=step,time=step*dt)
                print('PROGRESS',step,'change',err,'stable',stable,'min_g',float(g.min()),flush=True)
            if step%c['checkpoint_every']==0 or STOP or time.monotonic()-started>3000:
                checkpoint(path,g,h,step,signature,prev,stable)
                if STOP or time.monotonic()-started>3000:
                    (ROOT/'CONTINUE.json').write_text(json.dumps({'step':step,'signature':signature}));print('CHUNK_SAVED',step,flush=True);return
        checkpoint(path,g,h,step,signature,prev,stable)
        y=fields(g,h,v,w).cpu().numpy();np.savez(ROOT/'final.npz',x=x.cpu().numpy(),fields=y,columns=['rho','u','T','q','sig'],step=step,time=step*dt)
        result={'step':step,'time':step*dt,'stationary':stable>=c['stationarity_checks'],'reason':'stationarity' if stable>=c['stationarity_checks'] else 'time_limit_diagnostic','source':sha(__file__),'config':sha(ROOT/'config.json'),'checkpoint_sha256':sha(path),'final_sha256':sha(ROOT/'final.npz'),'scientific_validation':False}
        (ROOT/'RUN_FINISHED.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);c=json.loads((ROOT/'config.json').read_text());mode=sys.argv[1];device='cpu' if mode=='cpu-gate' else 'cuda'
    if device=='cuda': assert torch.cuda.is_available()
    if mode.endswith('gate'): gate(c,device)
    else: run(c,device)
