"""Independent domain/source audit and descriptive shape/implicit-matrix probes.

No solution changes, smoothing, eigenvalue regularization, or acceptance changes.
Critical determinant sign changes are diagnostics, not nonexistence/subshock proof.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.linalg import eig
from scipy.signal import find_peaks
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'stf_repaired_tensor'))
import shock_pilot as tm
import r26_shakhov_tensor as ts
sys.path.insert(0,str(H/'physics'))
import maxwell_repaired_mixed as sm
import projected_mixed as ss
F=['rho','u','theta','sigma','q','m','R','Delta','S']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def phase_x(a):
 x=a['x'];r=a['rho'];i=np.flatnonzero((r[:-1]-2)*(r[1:]-2)<=0);i=i[np.argmin(abs(x[i]))]
 return x-(x[i]+(2-r[i])*(x[i+1]-x[i])/(r[i+1]-r[i]))
def shape(a):
 x=phase_x(a);r={}
 for name,key in [('temperature','theta'),('density','rho')]:
  d=np.gradient(a[key],a['x'],edge_order=2);mask=(x>-15)&(x<12)
  ix=find_peaks(d,prominence=.01)[0];ix=ix[mask[ix]]
  r[name+'_slope_peaks']=[dict(x=float(x[i]),slope=float(d[i]),field=float(a[key][i])) for i in ix]
 return r
out={'domain_cases':{},'shape':{},'implicit_matrix':{},'interpretation':'Descriptive source/domain/grid diagnostics; determinant sign change alone does not prove a subshock, nonexistence, uniqueness, or the cause of profile shoulders.'}
stfh=json.loads((H/'stf_repair_hashes.json').read_text())
assert all(sha(H/k)==v for k,v in stfh.items())
out['independent_tensor_source_hashes']=stfh
out['independent_tensor_source_hashes']['stf_repaired_tensor/r26_shakhov_tensor.py']=sha(H/'stf_repaired_tensor/r26_shakhov_tensor.py')
for model,small,large,n,tens,scal in [('maxwell','maxwell_stf7681','maxwell_domain60',10241,tm,sm),('shakhov','projected15361','shakhov_domain60',20481,ts,ss)]:
 a=np.load(H/large/'profile.npz',allow_pickle=False);b=np.load(H/small/'profile.npz',allow_pickle=False);p=tens.Shock(n,60,3)
 t,rr,eq=p.evaluate(a['z']);f=p.residual(a['z']);vals=[t.rho,t.velocity[:,0],t.theta,t.sigma[:,0,0],t.heat_flux[:,0],t.m[:,0,0,0],t.R[:,0,0],t.Delta,t.R[:,0,0]+t.Delta/3]
 rec=max(float(np.max(abs(a[k]-v))) for k,v in zip(F,vals))
 cfg=json.loads((H/large/'manifest.json').read_text())['config']
 hashok=all(sha(H/k)==v for k,v in cfg['source_hashes'].items());finite=all(np.isfinite(a[k]).all() for k in a.files)
 coord=bool(np.array_equal(a['x'],p.x));pde=float(np.max(abs(eq)/p.eqscale));bc=float(max(abs(f[-11:])));pressure=float(min(np.min(t.rho*t.theta+t.sigma[:,0,0]),np.min(t.rho*t.theta-t.sigma[:,0,0]/2)))
 common=(a['x']>=-45)&(a['x']<=45);assert np.array_equal(a['x'][common],b['x'])
 differences={k:float(max(abs(a[k][common]-b[k]))) for k in F}
 record=dict(model=model,N=n,L=60,profile_sha=sha(H/large/'profile.npz'),source_hashes_match=hashok,finite=finite,coordinates_match=coord,nine_field_reconstruction=rec,independent_full_tensor_PDE=pde,boundary_phase=bc,minpressure=pressure,domain_field_max_difference=differences)
 record['independent_numerical_pass']=bool(hashok and finite and coord and rec==0 and pde<.01 and bc<1e-4 and pressure>0)
 out['domain_cases'][large]=record
 out['shape'][model]=shape(b)
 # Constant upstream generalized eigenproblem; no explicit A inverse.
 pp=tens.Shock(3,1,3);y=np.r_[pp.left/pp.scale,[0.,0.,0.]];D=np.r_[np.ones(5),pp.u1**2,pp.u1**3,pp.u1**3];zero=scal.point_rows(pp,y,np.zeros(8))[0]
 A=np.column_stack([scal.point_rows(pp,y,np.eye(8)[k]*D[k])[0]-zero for k in range(8)])
 eps=1e-6;B=np.column_stack([(scal.point_rows(pp,y+eps*np.eye(8)[k]*D[k],np.zeros(8))[0]-scal.point_rows(pp,y-eps*np.eye(8)[k]*D[k],np.zeros(8))[0])/(2*eps) for k in range(8)])
 lam=eig(-B,A,right=False);finiteeig=lam[np.isfinite(lam)];positive=sorted(float(v.real) for v in finiteeig if v.real>1e-8 and abs(v.imag)<1e-6)
 # Detect sign changes of scaled derivative matrix along saved mixed-state profile.
 Y=b['Y'];row=np.r_[p.eqscale[:2],[p.u1**2,p.u1**3,p.u1**3],p.eqscale[2:]]
 zero=scal.point_rows(p,Y,np.zeros_like(Y));AA=np.stack([(scal.point_rows(p,Y,np.broadcast_to(np.eye(8)[k]*D[k],Y.shape))-zero)/row for k in range(8)],axis=-1)
 signs,logdet=np.linalg.slogdet(AA);ix=np.flatnonzero(signs[:-1]*signs[1:]<0);x=phase_x(b)
 out['implicit_matrix'][model]=dict(upstream_positive_real_eigenvalues=positive,sign_change_brackets=[[float(x[i]),float(x[i+1])] for i in ix],dimensionless_scaling=dict(columns=D.tolist(),rows=row.tolist()))
old=np.load(H.parent/'r26_M3_discretization_repair_20261008/mixed7681/profile.npz');out['shape']['previous_Maxwell']=shape(old)
previous=json.loads((H.parent/'r26_M3_independent_physics_audit_20261008/upstream_modes.json').read_text())
oldlam=sorted(v[0] for v in previous['1e-06']['eigenvalues'] if np.isfinite(v[0]) and v[0]>1e-8 and abs(v[1])<1e-6)
out['Maxwell_upstream_vs_previous']=dict(previous_positive=oldlam,corrected_positive=out['implicit_matrix']['maxwell']['upstream_positive_real_eigenvalues'],note='Finite-difference eigenvalues: nonlinear STF repairs do not change equilibrium linearization; compare numerically, not bitwise across different evaluators.')
(H/'domain_and_shape_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
