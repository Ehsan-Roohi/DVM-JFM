"""Frozen bounded four-model shock stages; numerical tolerance is not model error."""
import argparse, json, hashlib, time, signal, os
from pathlib import Path
import numpy as np
import shakhov_reference as ref
import esbgk, fp, native_kernel, bgk_reference
P=Path(__file__).resolve().parent
STOP=False
def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
def dump(path,obj):ref.dump(path,obj)
def cases():return json.loads((P/'manifest.json').read_text())['cases']
def bindings():
 names=['matrix.py','manifest.json','esbgk.py','fp.py','shakhov_reference.py','native_kernel.py','bgk_reference.py','bgk_kernel.c','fp_kernel.c','kernel.so','fp_kernel.so']
 return {n:ref.sha(P/n) for n in names}
def setup(c):
 if c['model']!='ESFP':return ref.setup(c)
 x,v,w,f,l,r=fp.setup(c)
 # Isotropic positive starting guess, not a mixture with a potentially inadmissible FP tensor.
 a=ref.moments(f,v,w)
 f=np.ascontiguousarray(ref.eq(v,a[:,0],a[:,1],a[:,2]))
 return x,v,w,f,l,r
def inv(f,v,w):return esbgk.raw_invariants(f,v,w)
def step(f,v,w,l,r,c,dt):
 model=c['model']
 if model=='BGK':new=native_kernel.advance(f,v,w,l,r,c,dt);diag={}
 elif model=='ESFP':
  new,d,balance=fp.advance(f,v,w,l,r,c,dt)
  diag=dict(collision_invariant=float(d[4]),diffusion_min=float(d[3]),diffusion_fit=float(d[2]))
 elif model=='ESBGK':new=esbgk.advance(f,v,w,l,r,c,dt);diag={}
 else:new=ref.advance(f,v,w,l,r,c,dt);diag={}
 if not np.isfinite(new).all() or np.min(new)<0:raise FloatingPointError('Nonfinite or negative reduced distribution; no clipping')
 a=ref.moments(new,v,w)
 if not np.isfinite(a).all() or np.min(a[:,0])<=0 or np.min(a[:,2])<=0:raise FloatingPointError('Nonpositive rho/T')
 cx=v[None,:]-a[:,1,None]
 theta_x=np.sum(new[0]*cx**2*w,axis=1)/a[:,0]
 theta_y=np.sum(new[1]*w,axis=1)/(2*a[:,0])
 if min(np.min(theta_x),np.min(theta_y))<=0:raise FloatingPointError('Nonpositive gas covariance')
 face=np.where((v>=0)[None,None,:],np.concatenate([l[:,None,:],f],axis=1),np.concatenate([f,r[:,None,:]],axis=1))
 flux=inv(face*v[None,None,:],v,w)
 before=inv(f,v,w).sum(axis=0)
 bal=inv(new,v,w).sum(axis=0)-before+dt/(2*c['L']/c['nx'])*(flux[-1]-flux[0])
 defect=float(np.max(abs(bal)/np.maximum(abs(before),1)))
 if defect>5e-11:raise FloatingPointError(f'Finite-volume conservation failed: {defect}')
 diag['global_balance']=defect
 return new,diag
def gate(index):
 c=cases()[index];b=bindings();out=P/'gates'/c['name'];out.mkdir(parents=True,exist_ok=True)
 try:
  cc=dict(c,nx=24);x,v,w,f,l,r=setup(cc);dt=cc['CFL']*2*cc['L']/cc['nx']/cc['vmax']
  # A short physical-map test, including positivity and exact reservoir balance.
  worst=0.;start=time.monotonic();baseline=f.copy()
  for _ in range(20):f,d=step(f,v,w,l,r,cc,dt);worst=max(worst,d['global_balance'])
  restart=f.copy();ref.save(out/'restart.npz',f=f)
  aa,_=step(restart,v,w,l,r,cc,dt);bb,_=step(np.load(out/'restart.npz')['f'],v,w,l,r,cc,dt)
  assert np.array_equal(aa,bb)
  if c['model']=='BGK':
   old=baseline.copy()
   for _ in range(20):old=bgk_reference.advance(old,v,w,l,r,cc,dt)
   error=float(np.max(abs(old-f)/np.maximum(abs(old),1)))
   assert error<1e-11,error
  else:error=None
  # Collision identities use pre-collision primitive moments and implicit post stress.
  m=ref.moments(baseline,v,w);small=.01
  if c['model']=='ESBGK':
   z=esbgk.collision(baseline,v,w,c,small);n=ref.moments(z,v,w);rr=small*m[:,0]/1.5
   expect=np.column_stack([m[:,4]/(1+1.5*rr),m[:,3]/(1+rr),(m[:,6]+rr*.75*n[:,4]**2/n[:,0])/(1+rr)])
   actual=n[:,[4,3,6]];err=float(np.max(abs(actual-expect)/np.maximum(abs(expect),1)))
   assert err<1e-9,err
  elif c['model']=='ESFP':
   z,_=fp.collision(baseline,v,w,small);n=ref.moments(z,v,w);rr=small*m[:,0]/4.5
   expect=np.column_stack([m[:,4]/(1+4.5*rr),m[:,3]/(1+3*rr),(m[:,6]-15*rr*n[:,4]**2/n[:,0])/(1+4*rr)])
   err=float(np.max(abs(n[:,[4,3,6]]-expect)/np.maximum(abs(expect),1)))
   # These moment sources have axial velocity discretization error; not exact continuum identities.
   assert err<.002,err
  else:err=None
  dump(out/'PASS.json',dict(PASS=True,config=c,binding=b,twenty_step_seconds=time.monotonic()-start,max_global_balance=worst,native_reference_error=error,collision_source_error=err,restart_bitwise=True,scope='Short-map and collision oracle only; no stationarity or grid qualification'))
 except BaseException as e:
  dump(out/'FAILED.json',dict(PASS=False,config=c,binding=b,error=repr(e)));raise
def budgets(f,x,v,w,c,out):
 if c['model']=='ESFP':return fp.budgets(f,x,v,w,c,out)
 if c['model'] in ['BGK','ESBGK']:return esbgk.budgets(f,x,v,w,c,out)
 # Same flux channel, different heat-flux relaxation rate for Shakhov.
 a=ref.moments(f,v,w);rho,u,T,q,sig,R,D,S=a.T;cx=v[None,:]-u[:,None];g,h,k=f;tau=1/rho
 F=.5*np.sum(v*w*(cx**3*g+cx*h),axis=1);J=np.sum(v*w*(cx**4*g+2*cx**2*h+k),axis=1)
 L=np.sum(v*w*(1.5*cx**2*g+.5*h),axis=1);du=np.gradient(u,x,edge_order=2)
 Qq=-c['Pr']*q/tau;Qd=-D/tau
 integrand=Qq-du*L;Fb=F[0]+np.r_[0,np.cumsum(.5*(integrand[1:]+integrand[:-1])*np.diff(x))]
 Sb=2*(Fb-u*q)-5*rho*T*T-7*T*sig;Db=-tau*(np.gradient(J,x,edge_order=2)+8*du*F);Rb=Sb-Db/3
 mask=abs(S)>.02*np.max(abs(S));err=lambda p,t:float(np.linalg.norm((p-t)[mask])/max(np.linalg.norm(t[mask]),1e-30))
 report=dict(S_relative_L2=err(Sb,S),Delta_relative_L2=err(Db,D),R_relative_L2=err(Rb,R),note='Continuum steady-form budget; not stationarity proof or independent exact recovery.')
 ref.save(out/'budgets.npz',x=x,direct=a,S_reconstructed=Sb,Delta_reconstructed=Db,R_reconstructed=Rb,mask=mask,Qq=Qq,QDelta=Qd)
 dump(out/'BUDGETS.json',report);return report
def run(index):
 c=cases()[index];assert c['launch'],'Existing exclusive owner retained'
 gatefile=P/'gates'/c['name']/'PASS.json';g=json.loads(gatefile.read_text());assert g['PASS'] and g['binding']==bindings()
 out=P/'runs'/c['name'];out.mkdir(parents=True,exist_ok=True)
 signature=hashlib.sha256(json.dumps(dict(config=c,binding=g['binding']),sort_keys=True).encode()).hexdigest()
 x,v,w,f,l,r=setup(c);dt=c['CFL']*2*c['L']/c['nx']/c['vmax'];every=max(1,round(2/dt));stepno=stable=0;prev=None;drift=None;cp=out/'checkpoint.npz'
 if cp.exists():
  z=np.load(cp);assert str(z['signature'])==signature
  f=z['f'];stepno=int(z['step']);stable=int(z['stable']);prev=z['prev'];prev=None if not prev.size else prev
 start=time.monotonic();worst=0.;reason='physical_time_limit';minimum_pi=float('inf')
 dump(out/'START.json',dict(config=c,pid=os.getpid(),signature=signature,dt=dt,initial_guess='local isotropic Maxwellian' if c['model']=='ESFP' else 'positive RH mixture'))
 def save():ref.save(cp,f=f,step=stepno,stable=stable,prev=np.array([]) if prev is None else prev,signature=signature)
 try:
  while stepno*dt<c['tmax'] and stable<5 and not STOP:
   fn,d=step(f,v,w,l,r,c,dt);worst=max(worst,d['global_balance']);minimum_pi=min(minimum_pi,d.get('diffusion_min',float('inf')))
   if (stepno+1)%every==0:
    a=ref.moments(fn,v,w);old=ref.moments(f,v,w);scale=np.maximum(np.max(abs(a),axis=0),1e-8)
    rhs=float(np.max(abs(a-old)/(dt*scale)));drift=None if prev is None else float(np.max(abs(a-prev)/scale))
    stable=stable+1 if drift is not None and drift<c['tolerance'] and rhs<c['tolerance'] else 0;prev=a.copy()
    rec=dict(step=stepno+1,t=(stepno+1)*dt,rhs=rhs,drift=drift,stable=stable,tolerance=c['tolerance'],seconds=time.monotonic()-start)
    dump(out/'PROGRESS.json',rec);print(json.dumps(rec),flush=True)
   f=fn;stepno+=1
   if stepno%1000==0 or STOP:save()
   if time.monotonic()-start>c['wall_budget']:reason='cooperative_wall_budget';break
  if stable>=5:reason='stationary_checks_passed'
  elif STOP:reason='signal_checkpoint'
  save();a=ref.moments(f,v,w);extra,_=step(f,v,w,l,r,c,dt);rhs=float(np.max(abs(ref.moments(extra,v,w)-a)/(dt*np.maximum(np.max(abs(a),axis=0),1e-8))))
  ref.save(out/'profiles.npz',x=x,fields=a,columns=np.array(['rho','u','T','q','sigma','R','Delta','S']))
  b=budgets(f,x,v,w,c,out)
  report=dict(config=c,signature=signature,step=stepno,time=stepno*dt,reason=reason,stationary=stable>=5 and rhs<c['tolerance'],stable=stable,final_rhs=rhs,last_drift=drift,max_global_balance=worst,min_reduced=float(np.min(f)),min_diffusion_eigenvalue=None if not np.isfinite(minimum_pi) else minimum_pi,budgets=b,seconds=time.monotonic()-start,sha256={n:ref.sha(out/n) for n in ['checkpoint.npz','profiles.npz','budgets.npz']})
  dump(out/'STATUS.json',report);print(json.dumps(report),flush=True)
 except BaseException as e:
  save();dump(out/'FAILED.json',dict(error=repr(e),last_completed_step=stepno,checkpoint_sha256=ref.sha(cp)));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--gate',action='store_true');a=ap.parse_args()
 gate(a.index) if a.gate else run(a.index)
