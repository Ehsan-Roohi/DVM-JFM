"""Generate fast scalar evaluator of the repaired printed Maxwell R26 model."""
from pathlib import Path
import sys,json
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'stf_repaired_tensor'))
import shock_pilot
from r26_tensor_closures import stf2_project,stf3_project,stf4_project
t=shock_pilot.tensors_from_components(np.ones(1),np.ones(1),np.ones(1),np.ones(1),np.ones(1),np.ones(1),np.ones(1),np.ones(1))
p4=float(stf4_project(np.einsum('...ab,...cd->...abcd',t.sigma,t.sigma))[0,0,0,0,0])
p6=float(stf3_project(np.einsum('...ea,...bce->...abc',t.sigma,t.m))[0,0,0,0])
p7=float(stf3_project(np.einsum('...a,...bc->...abc',t.heat_flux,t.sigma))[0,0,0,0])
src=(HERE/'physics/projected_mixed.py').read_text(encoding='utf-8')
src=src.replace('Shakhov R26 from direct Grad26/f5 polynomial projection, isolated variant.','Printed Maxwell R26 with three STF bracket repairs; independent scalar projection.')
src=src.replace('Regularized outside moments use first-order GMM projection, all rates1/tau.','Original Maxwell molecule collision coefficients and nonlinear productions retained.')
src=src.replace('The exact Shakhov target, Pr2/3, mu=T and RH/phase constraints are fixed.','Pr2/3, mu=T and RH/phase constraints are fixed.')
old='calc=pt.constitutive(rho,u,T,sig,q,m,R,D,du,dT,dm,dR,dD)'
new=f'''calc=pt.constitutive(rho,u,T,sig,q,m,R,D,du,dT,dm,dR,dD)
    calc[:,0]=(calc[:,0]-.291*{p4!r}*sig*sig/rho)/2.097
    calc[:,1]=(calc[:,1]-1.203*{p6!r}*sig*m/rho-.854*{p7!r}*q*sig/rho)/1.698
    calc[:,2]-=2*(14*q*sig+7.5*m*sig)/(15*rho)'''
assert old in src;src=src.replace(old,new)
old='rates=np.column_stack([rho*sig,2*rho*q/3,rho*m,rho*R,rho*D])'
new='rates=np.column_stack([rho*sig,2*rho*q/3,1.5*rho*m,7*rho*R/6+sig*sig/3,2*rho*D/3+sig*sig])'
assert old in src;src=src.replace(old,new)
(HERE/'physics/maxwell_repaired_mixed.py').write_text(src,encoding='utf-8')
(HERE/'maxwell_scalar_coefficients.json').write_text(json.dumps(dict(STF4_sigma_sigma=p4,STF3_sigma_m=p6,STF3_q_sigma=p7),indent=2),encoding='utf-8')
print(p4,p6,p7)
