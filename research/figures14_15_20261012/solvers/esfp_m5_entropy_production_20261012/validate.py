import numpy as np,json,time,fp,shakhov_reference as ref
from pathlib import Path
p=Path(__file__).resolve().parent;rows=[]
def record(row):
 rows.append(row);ref.dump(p/'VALIDATION.json',dict(PASS=False,rows=rows));print(json.dumps(row),flush=True)
for nv in [800,1600,3200]:
 c=dict(mach=5,nx=12,L=30,nv=nv,vmax=40);x,v,w,f,l,r=fp.setup(c);a=ref.moments(f,v,w);dt=.001
 n,d=fp.collision(f,v,w,dt);b=ref.moments(n,v,w);anis=np.maximum(a[:,4]/a[:,0],-a[:,4]/(2*a[:,0]));beta=np.maximum(-1.25,-(1-1e-12)*a[:,2]/np.maximum(anis,1e-300));tau=2*(1-beta)/a[:,0];rr=dt/tau
 eq=ref.eq(v,np.ones(12),np.ones(12)*2,np.ones(12));en,ed=fp.collision(eq,v,w,dt)
 eqerr=float(np.max(abs(en-eq)));inv=float(np.max(abs(b[:,:3]-a[:,:3])/np.maximum(1,abs(a[:,:3]))));stress=float(np.max(abs(b[:,4]-a[:,4]/(1+2*(1-beta)*rr))))
 qerr=float(np.max(abs(b[:,3]-a[:,3]/(1+3*rr)))/max(1e-12,np.max(abs(a[:,3]))));Dexp=(a[:,6]+12*beta*rr*b[:,4]**2/a[:,0])/(1+4*rr);Derr=float(np.max(abs(b[:,6]-Dexp))/max(1e-12,np.max(abs(a[:,6]))))
 real=float(np.min(n[0]*n[2]-n[1]**2));record(dict(nv=nv,eqerr=eqerr,invariant=inv,stress=stress,qerror=qerr,Deltaerror=Derr,realizability=real,projection=d[2]))
 assert eqerr<1e-10 and inv<2e-11 and stress<1e-9 and real>=-1e-12
assert rows[-1]['qerror']<.002 and rows[-1]['Deltaerror']<.002
assert rows[-1]['qerror']<rows[0]['qerror'] and rows[-1]['Deltaerror']<rows[0]['Deltaerror']
z=np.load(p/'refined_seed_M5.npz',allow_pickle=False);c=json.loads(str(z['config']));f=z['f'];x,v,w,_,l,r=fp.setup(c);dt=c['CFL']*2*c['L']/c['nx']/c['vmax'];t=float(z['time'])
for i in range(2):
 a=ref.moments(f,v,w);n,d,bal=fp.advance(f,v,w,l,r,c,dt);b=ref.moments(n,v,w);real=float(np.min(n[0]*n[2]-n[1]**2));assert real>=-1e-12
 record(dict(failed_seed_step=i,time=t+dt,balance=bal,projection=d[2],minimum_diffusion=d[3],RHS=float(np.max(abs(b-a)/(dt*np.maximum(1e-8,np.max(abs(b),axis=0))))),realizability=real));f=n;t+=dt
 ref.save(p/'validated_checkpoint.npz',f=f,config=np.array(json.dumps(c)),step=int(z['step'])+i+1,time=t)
bad=f[:,:1,:].copy();bad[1]*=-1e3
try:fp.collision(bad,v,w,dt)
except FloatingPointError:record(dict(bad_energy_hard_stop=True))
else:raise AssertionError('bad energy was accepted')
ref.dump(p/'VALIDATION.json',dict(PASS=True,rows=rows,binding={n:ref.sha(p/n) for n in ['continue.py','fp.py','fp_kernel.c','fp_kernel.so','shakhov_reference.py','resource_watchdog.py']},validator_sha=ref.sha(p/'validate.py'),scope='Positive entropy moment projection, unchanged physical diffusion; seed continuation only, not convergence'))
