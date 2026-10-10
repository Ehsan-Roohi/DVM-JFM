"""Test Claude's critical compatibility proposal without changing any solution.

Sign convention explicitly A Xprime + b = 0. Cubic interpolation only for
read-only evaluation at critical locations, never for displayed profile smoothing.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'physics'))
import maxwell_repaired_mixed as mw
import projected_mixed as sk
def adj(A):
 n=A.shape[0];out=np.empty_like(A)
 for i in range(n):
  for j in range(n):out[j,i]=(-1)**(i+j)*np.linalg.det(np.delete(np.delete(A,i,axis=0),j,axis=1))
 return out
def probe(mat,y,deriv):
 A,b=mat(y);U,S,V=np.linalg.svd(A);ell=U[:,-1];r=V[-1];part=np.linalg.lstsq(A,-b,rcond=1e-10)[0]
 def directional(v):
  Ac,bc=mat(y.astype(complex)+1e-20j*v)
  return Ac.imag/1e-20,bc.imag/1e-20
 Ar,br=directional(r);Ap,bp=directional(part)
 coeff=np.array([ell@Ar@r,ell@(Ar@part+Ap@r+br),ell@(Ap@part+bp)])
 coef=coeff.copy()
 while len(coef)>1 and abs(coef[0])<1e-13*max(1,float(max(abs(coeff)))):coef=coef[1:]
 roots=np.roots(coef);alpha=float(r@(deriv-part));realroots=roots[abs(roots.imag)<1e-7].real
 def field(y):
  A,b=mat(y);return -adj(A)@b
 J=np.column_stack([field(y.astype(complex)+1e-20j*np.eye(len(y))[k]).imag/1e-20 for k in range(len(y))])
 eig=np.linalg.eigvals(J);idx=np.argsort(abs(eig))[::-1];eigs=eig[idx];two=eigs[:2]
 typ='undetermined'
 if max(abs(two.imag))<1e-6*max(1,max(abs(two))):typ='node' if np.prod(two.real)>0 else 'saddle'
 record=dict(compatibility_relative=float(abs(ell@b)/max(np.linalg.norm(b),1e-30)),slope_polynomial_coefficients=coeff.tolist(),alpha_saved=alpha,alpha_roots=[[float(v.real),float(v.imag)] for v in roots],distance_to_real_slope_root=float(min(abs(realroots-alpha))) if len(realroots) else None,desingularized_eigenvalues=[[float(v.real),float(v.imag)] for v in eigs],transverse_classification=typ,other_eigenvalues_max=float(max(abs(eigs[2:]))) if len(eigs)>2 else 0,minimum_singular_value=float(S[-1]))
 return record
tests={}
for name,k in [('node',2.),('saddle',-2.)]:
 def mat(y):return np.diag([1.+0*y[0],y[0]]),np.array([-1.+0*y[0],-k*y[1]])
 rec=probe(mat,np.zeros(2),np.array([1.,0.]));assert rec['transverse_classification']==name
 assert np.allclose(sorted(v[0] for v in rec['desingularized_eigenvalues']),sorted([1,k]))
 tests[name]=rec
def nozzle(y):return np.diag([1.+0*y[0],y[1]]),np.array([-1.+0*y[0],-y[0]])
rec=probe(nozzle,np.zeros(2),np.ones(2));assert np.allclose(sorted(v[0] for v in rec['alpha_roots']),[-1,1]);tests['two_analytic_slope_roots']=rec
out=dict(toy_gates=tests,models={},limitations='Cubic interpolation of finite-grid states; critical point not imposed during solve. Sign is A Xprime+b=0 (opposite b sign to Claude pseudocode). Classification and slope probes are diagnostic; no scientific acceptance or uniqueness conclusion. Full multidimensional tensor audit is outside the 1D shock scope.')
for model,sm,prefix,ns in [('maxwell',mw,'maxwell_stf',[961,1921,3841,7681]),('shakhov',sk,'projected',[961,1921,3841,7681,15361])]:
 cases=[]
 for n in ns:
  a=np.load(H/(prefix+str(n))/'profile.npz');p=sm.Shock(n,45,3);Y=a['Y'];x=a['x'];D=np.r_[np.ones(5),p.u1**2,p.u1**3,p.u1**3];row=np.r_[p.eqscale[:2],[p.u1**2,p.u1**3,p.u1**3],p.eqscale[2:]]
  def mat(X):
   yy=np.atleast_2d(X*D);b=sm.point_rows(p,yy,np.zeros_like(yy))/row
   A=np.stack([(sm.point_rows(p,yy,np.broadcast_to(np.eye(8)[k]*D[k],yy.shape))/row-b) for k in range(8)],axis=-1)
   return A[0],b[0]
  spline=CubicSpline(x,Y/D,axis=0);items=[]
  # Use independently measured linear bracket, then cubic-interpolant root.
  linear=json.loads((H/'critical_compatibility.json').read_text())['models'][model]
  roots=next(c['roots'] for c in linear if c['N']==n)
  for root in roots:
   i=np.searchsorted(x,root['x'])-1
   xc=brentq(lambda xx:float(np.linalg.det(mat(spline(xx))[0])),x[max(0,i-1)],x[min(n-1,i+2)],xtol=1e-13)
   rec=probe(mat,spline(xc),spline(xc,1));rec['x']=float(xc);items.append(rec)
  cases.append(dict(N=n,critical_points=items))
 out['models'][model]=cases
(H/'critical_slope_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({m:[dict(N=c['N'],critical=[{k:r[k] for k in ['x','compatibility_relative','alpha_saved','alpha_roots','distance_to_real_slope_root','transverse_classification','other_eigenvalues_max']} for r in c['critical_points']]) for c in cases] for m,cases in out['models'].items()},indent=2))
