"""Diagnose a sealed BGK state. Exact FV balance is not closure recovery."""
import argparse,json
from pathlib import Path
import numpy as np
import shakhov_reference as ref
import bgk_reference as old
P=Path(__file__).resolve().parent
W=P.parent
def main(index):
 seed=W/'bgk_fei_m2_wide_v1/runs/M2_velocity'
 status=json.loads((seed/'STATUS.json').read_text());c=status['config']
 assert status['stationary']
 assert ref.sha(seed/'checkpoint.npz')==status['sha256']['checkpoint.npz']
 x,v,w,_,left,right=ref.setup(c);z=np.load(seed/'checkpoint.npz');f=z['f'];dt=status['dt'];a=ref.moments(f,v,w)
 fn=old.advance(f,v,w,left,right,c,dt);b=ref.moments(fn,v,w)
 rhs=float(np.max(abs(b-a)/(dt*np.maximum(np.max(abs(a),axis=0),1e-8))))
 assert rhs<1e-6,rhs
 rho,u,T,q,sig,R,D,S=a.T;cx=v[None,:]-u[:,None]
 face=np.where((v>=0)[None,None,:],np.concatenate([left[:,None,:],f],axis=1),np.concatenate([f,right[:,None,:]],axis=1))
 fs=f-dt/(2*c['L']/c['nx'])*v[None,None,:]*(face[:,1:]-face[:,:-1])
 def project(y):
  g,h,k=y
  return np.column_stack([.5*np.sum(w*(cx**3*g+cx*h),axis=1),np.sum(w*(cx**4*g+2*cx**2*h+k),axis=1)])
 rate=project(fn-f)/dt;adv=project(f-fs)/dt;collision=project(fn-fs)/dt
 g,h,k=f;F=.5*np.sum(v*w*(cx**3*g+cx*h),axis=1);J=np.sum(v*w*(cx**4*g+2*cx**2*h+k),axis=1)
 L=np.sum(v*w*(1.5*cx**2*g+.5*h),axis=1);du=np.gradient(u,x,edge_order=2)
 cont=np.column_stack([np.gradient(F,x,edge_order=2)+du*L,np.gradient(J,x,edge_order=2)+8*du*F])
 E=ref.eq(v,rho,u,T);target=project(E)
 physical=np.column_stack([(target[:,0]-q)*rho,(target[:,1]-15*rho*T*T-D)*rho])
 temporal=-rate;spatial=cont-adv;splitting=collision-physical
 residual=cont-physical;closure=residual-temporal-spatial-splitting
 scale=np.maximum(np.linalg.norm(physical,axis=0)+np.linalg.norm(cont,axis=0),1e-30)
 report=dict(seed_sha256=ref.sha(seed/'checkpoint.npz'),source=ref.sha(__file__),original_numpy_final_rhs=rhs,channels=['heat','scalar_excess'],continuum_residual_L2=(np.linalg.norm(residual,axis=0)/scale).tolist(),temporal_L2=(np.linalg.norm(temporal,axis=0)/scale).tolist(),upwind_vs_continuum_L2=(np.linalg.norm(spatial,axis=0)/scale).tolist(),split_collision_vs_continuous_source_L2=(np.linalg.norm(splitting,axis=0)/scale).tolist(),decomposition_closure_L2=(np.linalg.norm(closure,axis=0)/scale).tolist(),note='Fixed pre-state peculiar-velocity projection. FV identity checks implementation; it does not certify independent recovery of R or Delta. Contributions may cancel.')
 assert np.max(np.linalg.norm(closure,axis=0)/scale)<1e-9
 ref.save(P/'BGK_M2_BUDGET_DECOMPOSITION.npz',x=x,residual=residual,temporal=temporal,spatial=spatial,splitting=splitting,closure=closure)
 ref.dump(P/'BGK_M2_BUDGET_DECOMPOSITION.json',report);print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);a=p.parse_args();main(a.index)
