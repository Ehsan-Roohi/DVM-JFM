"""Bounded moment-sufficient audit; no shock evolution or fullstate claim.

Cartesian Gaussian factorization computes target moments without an nx*nv
allocation. Historical four-iteration Newton and a 12-iteration sensitivity
are kept distinct. All input CSV quantities retain their original centre.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np

HERE=Path(__file__).resolve().parent
def evidence_path():
    for root in (HERE,*HERE.parents):
        for candidate in (root/'research/revision_runs/model_audit/evidence',root/'publication/research/revision_runs/model_audit/evidence'):
            if (candidate/'5.csv').is_file():return candidate
    raise FileNotFoundError('Archived heat-budget evidence missing')
INPUT=evidence_path()
CASES={2:(5,97,19,12.),3:(6,121,23,16.),5:(7,141,27,22.)}

def marginal(v,w,up,tp):
    a=np.exp(-(v[None,:]-up[:,None])**2/(2*tp[:,None]))*w
    return a/a.sum(axis=1)[:,None]

def target_moments(rho,u,T,nx,ny,vmax,niter,dtype):
    rho,u,T=[np.asarray(a,dtype=dtype) for a in (rho,u,T)]
    vx=np.linspace(-vmax,vmax,nx,dtype=dtype)
    vy=np.linspace(-vmax,vmax,ny,dtype=dtype)
    def weights(v):
        a=np.ones_like(v)*(v[1]-v[0]);a[[0,-1]]*=.5;return a
    wx,wy=weights(vx),weights(vy)
    up=u.copy();lt=np.log(np.maximum(T,1e-8))
    def stats(up,tp):
        px=marginal(vx,wx,up,tp);py=marginal(vy,wy,np.zeros_like(up),tp)
        ex=lambda a:np.sum(px*a,axis=1)
        ey=lambda a:np.sum(py*a,axis=1)
        mu=ex(vx);e2=ex(vx*vx)+2*ey(vy*vy)
        du=(vx[None,:]-up[:,None])/tp[:,None]
        dl=(vx[None,:]-up[:,None])**2/(2*tp[:,None])
        yl=vy[None,:]**2/(2*tp[:,None])
        dmdu=ex(vx*du)-mu*ex(du)
        dmdl=ex(vx*dl)-mu*ex(dl)
        dedu=ex(vx*vx*du)-ex(vx*vx)*ex(du)
        dedl=ex(vx*vx*dl)-ex(vx*vx)*ex(dl)+2*(ey(vy*vy*yl)-ey(vy*vy)*ey(yl))
        return px,py,mu,(e2-mu*mu)/3,dmdu,dmdl,(dedu-2*mu*dmdu)/3,(dedl-2*mu*dmdl)/3
    for _ in range(niter):
        _,_,mu,temp,a,b,c,d=stats(up,np.exp(lt))
        r1=mu-u;r2=temp-T;det=a*d-b*c
        det=np.where(np.abs(det)<1e-14,np.sign(det+1e-30)*1e-14,det)
        up+=np.clip((-r1*d+b*r2)/det,-.25,.25)
        lt+=np.clip((c*r1-a*r2)/det,-.25,.25)
    px,py,mu,temp,*_=stats(up,np.exp(lt))
    cx=vx[None,:]-u[:,None]
    ex=lambda a:np.sum(px*a,axis=1)
    ey2=np.sum(py*vy**2,axis=1);ey4=np.sum(py*vy**4,axis=1)
    q=.5*rho*(ex(cx**3)+2*ex(cx)*ey2)
    c4=rho*(ex(cx**4)+4*ex(cx**2)*ey2+2*ey4+2*ey2**2)
    return {'q':q,'delta':c4-15*rho*T*T,'mean_error':mu-u,'temperature_error':temp-T}

def cumtrap(y,x):
    out=np.zeros_like(y);out[1:]=np.cumsum(.5*(y[1:]+y[:-1])*np.diff(x));return out
def plateau(y):
    n=max(5,int(.08*len(y)));return y-np.mean(np.r_[y[:n],y[-n:]])
def norm(a,b,mask):return float(np.linalg.norm((a-b)[mask])/np.linalg.norm(b[mask]))
def smooth(a):return np.convolve(np.pad(a,(2,2),mode='edge'),np.array([1,4,6,4,1])/16,mode='valid')

results=[]
for mach,(e,nvx,nvy,vmax) in CASES.items():
    path=INPUT/f'{e}.csv';rows=list(csv.DictReader(path.open(encoding='utf-8')))
    d={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    x=d['x'];rho=d['rho_file'];u=d['ux_file'];T=d['T_file'];q=d['qx'];sig=d['sigma_xx']
    S=d['S_R_plus_Delta3_ref'];R=d['Rxx_closure'];D=d['Delta'];geom=d['geom_term'];F=d['Fq_direct']
    old=d['S_R_plus_Delta3_rec'];offset=F[0]-d['Fq_rec_from_balance'][0]
    target4=target_moments(rho,u,T,nvx,nvy,vmax,4,np.float64)
    target12=target_moments(rho,u,T,nvx,nvy,vmax,12,np.float64)
    target32=target_moments(rho,u,T,nvx,nvy,vmax,4,np.float32)
    # Only these two terms are changed: exact saved anchor, nonzero q[Md].
    anchored=old+2*offset
    corrected=anchored+2*cumtrap(target4['q'],x)
    # Original diagnostic smooths u for its derivative, but uses raw sig/m/q
    # in this integral. Preserve that convention and its archived ux centre.
    us=smooth(u)
    ux=np.gradient(us,x,edge_order=2)
    geom_true=ux*(u*(2.5*d['rho_from_f']*d['T_from_f']+sig)+d['mxxx']+2.2*q)
    corrected_geom=corrected-2*cumtrap(geom_true-geom,x)
    # Preserve plateau convention for comparisons to figures 2--4.
    mS=np.abs(plateau(S))>=.02*np.max(np.abs(plateau(S)))
    mR=np.abs(plateau(R))>=.02*np.max(np.abs(plateau(R)))
    metrics={}
    for name,arr in [('archived',old),('actual_anchor',anchored),('anchor_and_qMd',corrected),('anchor_qMd_and_geometry',corrected_geom)]:
        metrics[name]={'S_raw_active5_relL2':norm(arr,S,np.abs(S)>.05*np.max(np.abs(S))),
                       'S_plateau_active2_relL2':norm(plateau(arr),plateau(S),mS),
                       'R_given_archived_Delta_plateau_active2_relL2':norm(plateau(arr-D/3),plateau(R),mR)}
    res={'Mach':mach,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'status':'MOMENT_SUFFICIENT_CONDITIONAL_TARGET_NOT_FULLSTATE_REGENERATION',
         'nx':len(x),'velocity_grid':[nvx,nvy,nvy],'vmax':vmax,'tau_in_stored_x':1,
         'anchor_Fq_actual_minus_old':float(offset),'metrics':metrics,
         'qMd_max':float(np.max(np.abs(target4['q']))),
         'qMd_4_vs_12_max':float(np.max(np.abs(target4['q']-target12['q']))),
         'qMd_float32_vs64_max':float(np.max(np.abs(target32['q']-target4['q']))),
         'target4_mean_error_max':float(np.max(np.abs(target4['mean_error']))),
         'target4_temperature_error_max':float(np.max(np.abs(target4['temperature_error']))),
         'plateau_removed_anchor_effect_max':float(np.max(np.abs(plateau(anchored)-plateau(old))))}
    results.append(res)
    np.savez(Path(__file__).parent/f'M{mach}_budget_correction.npz',x=x,S_ref=S,S_archived=old,S_anchor=anchored,
             S_qMd=corrected,S_geometry=corrected_geom,qMd=target4['q'],DeltaMd=target4['delta'],geom=geom,geom_true=geom_true)
(Path(__file__).parent/'MOMENT_SUFFICIENT_BUDGET_CHECK.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
