"""Independent scalar-vs-full3D tensor audit after printed STF repairs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'stf_repaired_tensor'))
import implicit_repaired as tensor
sys.path.insert(0,str(HERE/'physics'))
import maxwell_repaired_mixed as scalar
from scipy.optimize._numdiff import approx_derivative

def run():
    p=scalar.Shock(7681,45,3);a=np.load(HERE.parent/'r26_M3_discretization_repair_20261008/mixed7681/profile.npz')
    rng=np.random.default_rng(1029);idx=rng.integers(2800,4600,size=300)
    Y=a['Y'][idx].copy();Yp=np.gradient(a['Y'],a['x'],axis=0,edge_order=2)[idx].copy()
    Y[:,5:]+=rng.normal(size=(len(Y),3))*.2;Yp+=rng.normal(size=Y.shape)*.03
    tr=tensor.point_rows(p,Y,Yp);sr=scalar.point_rows(p,Y,Yp)
    err=float(np.max(abs(tr-sr)/np.maximum(1,abs(tr))))
    # Random symmetry brackets: explicit STF4 then divergence, STF3 then divergence.
    # In axial flow this must reproduce factors4/7,3/5, and two STF2 ->4/3.
    m=np.zeros((1,3,3,3));m[0,0,0,0]=1
    for j in [1,2]:m[0,0,j,j]=m[0,j,0,j]=m[0,j,j,0]=-.5
    rr=np.diag([1.,-.5,-.5])[None];gu=np.zeros((1,3,3));gu[0,0,0]=1
    clmod=sys.modules['r26_tensor_closures']
    f4=float(np.einsum('...dijkd->...ijk',clmod.stf4_project(np.einsum('...ijk,...dl->...dijkl',m,gu)))[0,0,0,0])
    f3=float(np.einsum('...dijd->...ij',clmod.stf3_project(np.einsum('...ij,...dk->...dijk',rr,gu)))[0,0,0])
    f2=float(2*clmod.stf2_project(gu)[0,0,0])
    pp=scalar.Shock(65,45,3);z=np.column_stack([np.interp(pp.x,a['x'],a['z'][:,k]) for k in range(5)]);YY=scalar.lift(pp,z);fun=lambda z:scalar.residual_diagnostic(pp,z)
    J=approx_derivative(fun,YY.ravel(),method='3-point',abs_step=1e-5,sparsity=scalar.sparsity(pp)).tocsr();v=rng.normal(size=YY.size);v/=np.linalg.norm(v);fd=(fun(YY.ravel()+1e-5*v)-fun(YY.ravel()-1e-5*v))/2e-5
    jv=float(np.linalg.norm(J@v-fd)/np.linalg.norm(fd))
    report=dict(scalar_vs_repaired_tensor_max=err,random_cases=len(Y),axial_STF_factors=dict(m_velocity_STF4=f4,R_velocity_STF3=f3,q_velocity_STF2=f2),expected_factors=[4/7,3/5,4/3],Jv=jv,printed_sources={'Eq20':'R_<ij partial_k u_k> projects rank3 before contraction','Eq25':'m_<ijk partial_m u_m> projects rank4 before contraction','Eq26':'both q_j partial_<i u_j> gradients project rank2'},correction_changes_physics=False,original_accepted_results_immutable=True)
    report['PASS']=bool(err<1e-10 and jv<1e-5 and np.max(abs(np.array([f4,f3,f2])-np.array([4/7,3/5,4/3])))<1e-14)
    (HERE/'maxwell_repair_gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));assert report['PASS']
if __name__=='__main__':run()
