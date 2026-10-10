"""Freeze an isolated Shakhov R26 variant; never edit the Maxwell campaign."""
from pathlib import Path
import shutil, ast, hashlib, json
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SRC=ROOT/'revision_runs/r26_M3_discretization_repair_20261008/provenance'
PHYS=OUT/'physics';PHYS.mkdir(exist_ok=True)
for name in ['shock_pilot.py']:
    shutil.copyfile(SRC/name,PHYS/name)
(PHYS/'cavity_source').mkdir(exist_ok=True)
for name in ['r26_state.py','r26_tensor_closures.py','r26_bulk_equations.py']:
    shutil.copyfile(SRC/'cavity_source'/name,PHYS/'cavity_source'/name)
source=(SRC/'cavity_source/r26_tensor_closures.py').read_text(encoding='utf-8')
tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='closures_from_tensors')
body=ast.get_source_segment(source,fn).replace('def closures_from_tensors(','def closures_from_tensors(')
body=body.replace('coefficients = closure_coefficients(coefficient_mode)',"assert coefficient_mode == 'shakhov-derived'\n    coefficients = SimpleNamespace(C1=1., C2=0., Y1=1., Y2=0., Y3=0., mode='shakhov-derived', provenance='Shakhov projection + Gu Appendix A kinetic transport')")
body=body.replace('        + t8\n','        # No stress/heat or stress/m quadratic collision in Shakhov.\n')
assert '        + t8\n' not in body
header='''"""Shakhov collision + first-order Grad26 regularization, isolated diagnostic.
Derived from target polynomial and Gu Appendix A kinetic transport.
All outside-GMM collision rates are 1/tau. Pr=2/3, mu=theta.
Original frozen kernels remain available for explicit Maxwell comparisons.
"""
from types import SimpleNamespace
import dataclasses
import numpy as np
import shock_pilot as base
from shock_pilot import tensors_from_components, R26Gradients, R26ClosureDerivatives, farfields
from r26_state import StateTensors
from r26_tensor_closures import *
from r26_tensor_closures import _validate_tensors_and_gradients, _leading_shape, _broadcast_mu
'''
bulk='''
def steady_r26_bulk_residual(t,g,c,cd,*,mu=1.,acceleration=None):
    """Replace only analytically identified Maxwell collision productions."""
    r=base.steady_r26_bulk_residual(t,g,c,cd,mu=mu,acceleration=acceleration)
    nu=t.rho*t.theta/np.asarray(mu)
    sig2=stf2_project(np.einsum('...ki,...jk->...ij',t.sigma,t.sigma))
    norm2=np.einsum('...ij,...ij->...',t.sigma,t.sigma)
    return dataclasses.replace(r,
        m=r.m-.5*nu[...,None,None,None]*t.m,
        R=r.R-nu[...,None,None]*t.R/6-2*nu[...,None,None]*sig2/(3*t.rho[...,None,None]),
        Delta=r.Delta+nu*t.Delta/3-2*nu*norm2/(3*t.rho),
        provenance='Shakhov collision projection + unchanged kinetic R26 transport')

class Shock(base.Shock):
    def evaluate(self,z):
        v=np.asarray(z).reshape(self.n,5)*self.scale
        dv=np.gradient(v,self.x,axis=0,edge_order=2)
        t=self.fields(z);u=t.velocity[:,0];T=t.theta;rho=t.rho
        du=u*dv[:,0];dT=T*dv[:,1];drho=-rho*dv[:,0]
        ds=-self.mass*du-drho*T-rho*dT
        dq=-self.mass*(u*du+2.5*dT)-du*t.sigma[:,0,0]-u*ds
        dt=tensors_from_components(drho,du,dT,dq,ds,dv[:,2],dv[:,3],dv[:,4])
        def g(a):return np.stack([a,np.zeros_like(a),np.zeros_like(a)],axis=1)
        grad=R26Gradients(**{f.name:g(getattr(dt,f.name)) for f in dataclasses.fields(R26Gradients)})
        cl=closures_from_tensors(t,grad,mu=T,coefficient_mode='shakhov-derived')
        dphi=np.gradient(cl.phi,self.x,axis=0,edge_order=2)
        dpsi=np.gradient(cl.psi,self.x,axis=0,edge_order=2)
        do=np.gradient(cl.Omega,self.x,axis=0,edge_order=2)
        cd=R26ClosureDerivatives(dphi[...,0],dpsi[...,0],g(do))
        res=steady_r26_bulk_residual(t,grad,cl,cd,mu=T)
        eq=np.column_stack([res.stress[:,0,0],res.heat_flux[:,0],res.m[:,0,0,0],res.R[:,0,0],res.Delta])
        return t,res,eq
'''
(PHYS/'r26_shakhov.py').write_text(header+'\n'+body+'\n'+bulk,encoding='utf-8')
mixed=(ROOT/'revision_runs/r26_numerical_audit_20261008/mixed_prototype/implicit_mixed.py').read_text(encoding='utf-8')
mixed=mixed.replace("ROOT=Path(__file__).resolve().parents[2]/'r26_m2_pilot_20261007'/'numerical_repair_20261008'","ROOT=Path(__file__).resolve().parent")
mixed=mixed.replace('import shock_pilot as s','import r26_shakhov as s').replace("coefficient_mode='jfm2009'","coefficient_mode='shakhov-derived'")
(PHYS/'implicit_shakhov.py').write_text(mixed,encoding='utf-8')
files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in PHYS.rglob('*.py')}
(OUT/'source_hashes.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print(json.dumps(files,indent=2))
