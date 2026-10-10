"""Reconstruct continuum central moments and measured collision sources from exact raw polynomials."""
from pathlib import Path
import itertools,math,json,sys
import numpy as np
P=[(a,b,d-a-b) for d in range(6) for a in range(d+1) for b in range(d-a+1)]
IDX={v:i for i,v in enumerate(P)}
def centered(raw,u,abc):
 out=np.zeros(raw.shape[0])
 for a in range(abc[0]+1):
  for b in range(abc[1]+1):
   for c in range(abc[2]+1):
    out+=raw[:,IDX[a,b,c]]*math.comb(abc[0],a)*math.comb(abc[1],b)*math.comb(abc[2],c)*(-u[:,0])**(abc[0]-a)*(-u[:,1])**(abc[1]-b)*(-u[:,2])**(abc[2]-c)
 return out
def fields(raw,col):
 rho=raw[:,0];u=np.column_stack([raw[:,IDX[q]]/rho for q in [(1,0,0),(0,1,0),(0,0,1)]])
 C=lambda q:centered(raw,u,q)
 Q=lambda q:centered(col,u,q)
 c2=C((2,0,0))+C((0,2,0))+C((0,0,2));T=c2/(3*rho)
 c4=C((4,0,0))+C((0,4,0))+C((0,0,4))+2*(C((2,2,0))+C((2,0,2))+C((0,2,2)))
 c2x=C((3,0,0))+C((1,2,0))+C((1,0,2))
 c2xx=C((4,0,0))+C((2,2,0))+C((2,0,2))
 c4x=C((5,0,0))+C((1,4,0))+C((1,0,4))+2*(C((3,2,0))+C((3,0,2))+C((1,2,2)))
 sig=C((2,0,0))-rho*T;delta=c4-15*rho*T*T;R=c2xx-c4/3-7*T*sig
 qdelta=Q((4,0,0))+Q((0,4,0))+Q((0,0,4))+2*(Q((2,2,0))+Q((2,0,2))+Q((0,2,2)))
 return dict(rho=rho,u=u[:,0],uy=u[:,1],uz=u[:,2],theta=T,pxx=C((2,0,0)),pyy=C((0,2,0)),pzz=C((0,0,2)),sigma=sig,q=.5*c2x,m=C((3,0,0))-.6*c2x,R=R,Delta=delta,S=R+delta/3,phi=C((4,0,0))-6/7*c2xx+3/35*c4,C4=c4,C5x=c4x,Fq=.5*(u[:,0]*c2x+c2xx),JDelta=u[:,0]*c4+c4x,KDelta=u[:,0]*c2x+c2xx,Qq=.5*(Q((3,0,0))+Q((1,2,0))+Q((1,0,2))),QDelta=qdelta,Qsigma=Q((2,0,0))-(Q((2,0,0))+Q((0,2,0))+Q((0,0,2)))/3)
def load_block(f):
 h=np.fromstring(f.open().readline(),sep=' '); a=np.loadtxt(f,skiprows=1);assert a.shape==(800,127) and np.isfinite(a).all()
 return h,a

def process(folder):
 folder=Path(folder); data=[load_block(f) for f in sorted((folder/'blocks').glob('block_*.dat'))];late=[a for h,a in data if h[-1]==1]
 if not late:return {'sampled_blocks':0,'warmup_blocks':len(data)}
 avg=np.mean(late,axis=0);f=fields(avg[:,1:57],avg[:,57:92]);f['x']=avg[:,0]
 stab=fields(avg[:,1:57],avg[:,92:127]);f.update({k+'_stabilization':stab[k] for k in ['Qq','QDelta','Qsigma']})
 blockf=[fields(a[:,1:57],a[:,57:92]) for a in late]
 scatter={k:np.std([b[k] for b in blockf],axis=0,ddof=1) if len(late)>1 else np.full(800,np.nan) for k in blockf[0]}
 np.savez(folder/'profiles.npz',**f,**{k+'_block_sd':v for k,v in scatter.items()},raw_moments=avg[:,1:57],collision_raw=avg[:,57:92],stabilization_raw=avg[:,92:127])
 names=list(f);np.savetxt(folder/'profiles.csv',np.column_stack([f[k] for k in names]),delimiter=',',header=','.join(names),comments='')
 report={'sampled_blocks':len(late),'total_blocks':len(data),'stationarity_certified':False,'uncertainty':'Block standard deviation only, not autocorrelation-corrected confidence interval. Higher moment central estimates have finite sample bias.','finite':all(np.isfinite(v).all() for v in f.values()),'min_rho':float(f['rho'].min()),'min_theta':float(f['theta'].min()),'net_stabilization_displacement':float(sum(h[5] for h,a in data))}
 if len(late)>=4:
  mid=len(late)//2;fa=fields(np.mean(late[:mid],axis=0)[:,1:57],np.mean(late[:mid],axis=0)[:,57:92]);fb=fields(np.mean(late[mid:],axis=0)[:,1:57],np.mean(late[mid:],axis=0)[:,57:92]);report['half_window_relative_l2']={k:float(np.linalg.norm(fa[k]-fb[k])/max(np.linalg.norm(fb[k]),1e-30)) for k in fa}
 (folder/'analysis.json').write_text(json.dumps(report,indent=2));return report

def selftest():
 # Gauss-Hermite exact integration of polynomial Maxwellian, including nonzero 3D bulk.
 from numpy.polynomial.hermite import hermgauss
 z,w=hermgauss(5);pts=np.array(list(itertools.product(z,z,z)))*np.sqrt(2)*1.2+np.array([2.3,.2,-.1]);weights=np.prod(np.array(list(itertools.product(w,w,w))),axis=1)/np.pi**1.5
 raw=np.array([[np.sum(weights*np.prod(pts**np.array(p),axis=1)) for p in P]])
 f=fields(raw,np.zeros((1,35)))
 assert max(abs(f[k][0]) for k in ['q','m','R','Delta','S','phi','QDelta'])<1e-9
 assert abs(f['theta'][0]-1.44)<1e-12
 # Independent particle evaluation of core moments and budget fluxes.
 rng=np.random.default_rng(23);v=rng.normal(size=(10000,3))*[2,.9,1.3]+[3,.1,-.2]
 raw=np.array([[np.mean(np.prod(v**np.array(p),axis=1)) for p in P]]);f=fields(raw,np.zeros((1,35)));c=v-v.mean(axis=0);c2=(c*c).sum(axis=1);T=c2.mean()/3;sg=(c[:,0]**2).mean()-T
 direct={'q':np.mean(c2*c[:,0])/2,'Delta':np.mean(c2*c2)-15*T*T,'R':np.mean(c2*c[:,0]**2)-np.mean(c2*c2)/3-7*T*sg,'JDelta':np.mean(v[:,0]*c2*c2),'Fq':np.mean(v[:,0]*c2*c[:,0])/2}
 err=max(abs(f[k][0]-v) for k,v in direct.items());assert err<1e-9
 return {'maxwellian_and_direct_particle_pass':True,'max_abs_error':err}
if __name__=='__main__':print(json.dumps(process(sys.argv[1]) if len(sys.argv)>1 else selftest(),indent=2))
