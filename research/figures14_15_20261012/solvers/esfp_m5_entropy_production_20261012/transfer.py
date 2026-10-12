from pathlib import Path
p=Path(__file__).resolve().parent
src=p.parent/"esfp_m5_entropy_20261012"
import json
import numpy as np,sys
sys.path.insert(0,str(p));import shakhov_reference as ref
z=np.load(src/'validated_checkpoint.npz',allow_pickle=False);c=json.loads(str(z['config']));old=z['f'];dv=2*c['vmax']/c['nv'];v=-c['vmax']+(np.arange(c['nv'])+.5)*dv;w=np.full(c['nv'],dv);a=ref.moments(old,v,w)
f=old.reshape(3,c['nx'],c['nv']//4,4).mean(axis=3);c=dict(c,nv=c['nv']//4);dv=2*c['vmax']/c['nv'];v=-c['vmax']+(np.arange(c['nv'])+.5)*dv;w=np.full(c['nv'],dv)
base=f[0].copy();zz=(v[None,:]-a[:,1,None])/np.sqrt(a[:,2,None]);aa=np.zeros(c['nx']);bb=aa.copy();goal=1+a[:,4]/(a[:,0]*a[:,2])
for it in range(30):
 expo=aa[:,None]*zz+bb[:,None]*zz**2;assert np.isfinite(expo).all() and np.max(abs(expo))<600
 g=base*np.exp(expo);norm=g.sum(axis=1)*dv
 m=[(g*zz**j).sum(axis=1)*dv/norm for j in range(1,5)];m1,m2,m3,m4=m
 if max(np.max(abs(m1)),np.max(abs(m2-goal)))<2e-13:break
 A=m2-m1*m1;C=m3-m1*m2;E=m4-m2*m2;det=A*E-C*C;assert np.min(det)>1e-24
 aa+=(-m1*E+C*(m2-goal))/det;bb+=(-A*(m2-goal)+C*m1)/det
else:raise RuntimeError('transfer moment projection did not converge')
f[0]=g*(a[:,0]/norm)[:,None];b=ref.moments(f,v,w);errs=np.max(abs(b-a),axis=0)/np.maximum(1e-8,np.max(abs(a),axis=0))
assert np.isfinite(f).all() and np.min(f)>=0 and np.max(errs[:3])<2e-11 and np.max(errs[3:])<1e-4
assert not np.any(f[0]*f[2]<f[1]**2*(1-1e-12))
ref.dump(p/'VELOCITY_TRANSFER.json',dict(from_nv=51200,to_nv=c['nv'],relative_eight_field_changes=errs.tolist(),invariants_preserved=True,scope='checkpoint transfer only; velocity accuracy remains to qualify'))
ref.save(p/'refined_seed_M5.npz',f=f,config=np.array(json.dumps(c)),step=int(z['step']),time=float(z['time']))

