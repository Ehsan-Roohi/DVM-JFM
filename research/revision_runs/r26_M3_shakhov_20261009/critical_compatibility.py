"""Read-only critical-point compatibility across independently solved grids.

Linear interpolation of saved states; no new root solve for the shock, no BC deletion.
Compatibility normalisation is descriptive and has no acceptance threshold.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.optimize import brentq
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'physics'))
import maxwell_repaired_mixed as mw
import projected_mixed as sk
out={'models':{},'interpretation':'At each determinant sign crossing, ell^T b is a necessary smooth-crossing condition of A(Y)Yprime+b(Y)=0. Saved-state interpolation, not independently imposed critical matching. Small compatibility does not prove unique branch or physical validity. No new threshold.'}
for model,sm,prefix,ns in [('maxwell',mw,'maxwell_stf',[961,1921,3841,7681]),('shakhov',sk,'projected',[961,1921,3841,7681,15361])]:
 cases=[]
 for n in ns:
  a=np.load(H/(prefix+str(n))/'profile.npz');p=sm.Shock(n,45,3);Y=a['Y'];x=a['x'];row=np.r_[p.eqscale[:2],[p.u1**2,p.u1**3,p.u1**3],p.eqscale[2:]];D=np.r_[np.ones(5),p.u1**2,p.u1**3,p.u1**3]
  def mat(y):
   y=np.atleast_2d(y);b=sm.point_rows(p,y,np.zeros_like(y))/row
   A=np.stack([(sm.point_rows(p,y,np.broadcast_to(np.eye(8)[k]*D[k],y.shape))/row-b) for k in range(8)],axis=-1)
   return A,b
  AA,bb=mat(Y);sign=np.linalg.slogdet(AA)[0];ix=np.flatnonzero(sign[:-1]*sign[1:]<0);checks=[]
  for i in ix:
   def interp(f):return (1-f)*Y[i]+f*Y[i+1]
   f=brentq(lambda f:float(np.linalg.det(mat(interp(f))[0][0])),0,1,xtol=1e-13)
   yy=interp(f);A,b=mat(yy);A=A[0];b=b[0];U,S,V=np.linalg.svd(A);ell=U[:,-1]
   compat=float(abs(ell@b));ratio=compat/max(float(np.linalg.norm(b)),1e-30)
   deriv=(Y[i+1]-Y[i])/(x[i+1]-x[i]);res=A@(deriv/D)+b
   checks.append(dict(x=float(x[i]+f*(x[i+1]-x[i])),singular_values=S.tolist(),left_null_compatibility_absolute=compat,left_null_compatibility_relative=ratio,interpolated_full_point_residual_max=float(max(abs(res))),physical_dTdx=float(np.exp(yy[1])*deriv[1]),note='Finite-grid interpolation diagnostic; not a solution-error bound.'))
  cases.append(dict(N=n,dx=float(x[1]-x[0]),roots=checks))
 out['models'][model]=cases
(H/'critical_compatibility.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({m:[dict(N=c['N'],roots=[{k:v for k,v in r.items() if k in ['x','left_null_compatibility_absolute','left_null_compatibility_relative','physical_dTdx']} for r in c['roots']]) for c in cases] for m,cases in out['models'].items()},indent=2))
