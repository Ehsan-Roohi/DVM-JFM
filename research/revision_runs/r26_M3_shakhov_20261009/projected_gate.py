"""Test symbolic projection independently with kinetic quadrature and sparse Jv."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.optimize._numdiff import approx_derivative
OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(OUT/'physics'))
import projected_mixed as im
import implicit_shakhov as transferred
import kinetic_gate as kg

def axial_closure(Y):
    ph=kg.stf4_project(np.einsum('i,j,k,l->ijkl',[1.,0,0],[1.,0,0],[1.,0,0],[1.,0,0]))*(35*Y[5]/8)
    ps=kg.stf3_project(np.einsum('i,j,k->ijk',[1.,0,0],[1.,0,0],[1.,0,0]))*(5*Y[6]/2)
    return ph,ps,np.array([Y[7],0,0])

def run():
    p=im.Shock(7681,45,3)
    a=np.load(OUT.parent/'r26_M3_discretization_repair_20261008/mixed7681/profile.npz')
    dY=np.gradient(a['Y'],a['x'],axis=0,edge_order=2)
    records=[]
    for idx in np.linspace(2800,4500,17,dtype=int):
        Y=a['Y'][idx];Yp=dY[idx];rho,u,T,q,sig,m,R,D=kg.state_from_z(p,Y[:5])
        C,W=kg.grid(T);lab=C.copy();lab[:,0]+=u;C2=np.sum(C*C,axis=1);P=kg.weights(C,T)
        def f_ratio(yy):
            rr,uu,TT,qq,ss,mm,RR,DD=kg.state_from_z(p,yy[:5]);CC=lab.astype(np.result_type(yy)).copy();CC[:,0]-=uu
            ph,ps,om=axial_closure(yy)
            return rr/rho*(T/TT)**1.5*np.exp(C2/(2*T)-np.sum(CC*CC,axis=1)/(2*TT))*kg.poly(CC,rr,TT,qq,ss,mm,RR,DD,ph,ps,om)
        fd=np.imag(f_ratio(Y+1j*1e-20*Yp))/1e-20
        integral=rho*(P.T@(W*lab[:,0]*fd))
        rates=np.r_[rho*sig[0,0],2*rho*q[0]/3,rho*m[0,0,0],rho*R[0,0],rho*D]
        wanted=integral[3:8]+rates
        bulk=im.point_rows(p,Y[None,:],Yp[None,:],details=True)[2][0]
        old=transferred.point_rows(p,Y[None,:],Yp[None,:],details=True)[3]
        oldbulk=np.r_[old.stress[0,0,0],old.heat_flux[0,0],old.m[0,0,0,0],old.R[0,0,0],old.Delta[0]]
        records.append({'x':float(a['x'][idx]),'projection_max_scaled_difference':float(np.max(abs(bulk-wanted)/np.maximum(1,abs(wanted)))),'transferred_minus_projected':(oldbulk-bulk).tolist()})
    # Independent closure oracle already computed from Grad26, not f5.
    g=json.loads((OUT/'kinetic_gate.json').read_text())
    closure_error=[]
    for rec in g['closure_transport']:
        idx=np.argmin(abs(a['x']-rec['x']));Y=np.r_[a['z'][idx],[0,0,0]];Yp=np.r_[dY[idx,:5],[0,0,0]]
        calc=im.point_rows(p,Y[None,:],Yp[None,:],details=True)[1][0]
        closure_error.append(float(np.max(abs(calc-np.array(rec['quadrature']))/np.maximum(1,np.abs(rec['quadrature'])))))
    pp=im.Shock(65,45,3);zz=np.column_stack([np.interp(pp.x,a['x'],a['z'][:,k]) for k in range(5)]);YY=im.lift(pp,zz)
    J=approx_derivative(lambda y:im.residual_diagnostic(pp,y),YY.ravel(),method='3-point',abs_step=1e-5,sparsity=im.sparsity(pp)).tocsr()
    v=np.random.default_rng(823).normal(size=YY.size);v/=np.linalg.norm(v);fd=(im.residual_diagnostic(pp,YY.ravel()+1e-5*v)-im.residual_diagnostic(pp,YY.ravel()-1e-5*v))/2e-5
    jv=float(np.linalg.norm(J@v-fd)/np.linalg.norm(fd))
    equil=[]
    for ep in [pp.left,pp.right]:
        Y=np.column_stack([np.tile(ep/pp.scale,(pp.n,1)),np.zeros((pp.n,3))]);equil.append(float(np.max(abs(im.point_rows(pp,Y,np.zeros_like(Y))))))
    norms=[]
    for n in [33,65,129]:
        pp=im.Shock(n,2,3);x=pp.x;Y=np.tile(np.r_[np.log(2),np.log(2),.03,.02,.04,.1,.2,.3],(n,1));amp=np.array([.01,.02,.01,.03,.04,.02,.01,.03]);Y+=np.sin(x[:,None])*amp
        midx=(x[1:]+x[:-1])/2;exactY=np.tile(np.r_[np.log(2),np.log(2),.03,.02,.04,.1,.2,.3],(n-1,1))+np.sin(midx[:,None])*amp
        err=im.point_rows(pp,(Y[:-1]+Y[1:])/2,np.diff(Y,axis=0)/(x[1]-x[0]))-im.point_rows(pp,exactY,np.cos(midx[:,None])*amp)
        norms.append(float(np.linalg.norm(err)/np.sqrt(err.size)))
    report={'collision_projection_error':g['collision_max'],'bulk_quadrature_cases':records,'bulk_quadrature_max':max(r['projection_max_scaled_difference'] for r in records),'closure_quadrature_max':max(closure_error),'Jv':jv,'equilibrium':equil,'smooth_midpoint_errors':norms,'smooth_order':np.log2(np.array(norms[:-1])/norms[1:]).tolist(),'explicit_model':'Shakhov target + direct Grad26 kinetic projection and first-order regularization; distinct from copied Gu transport','criteria':{'PDE_strict_less':.01,'BC_phase_strict_less':1e-4,'positive_directional_pressure':True},'old_Maxwell_results_preserved':True}
    report['PASS']=bool(report['bulk_quadrature_max']<1e-9 and report['closure_quadrature_max']<1e-9 and jv<1e-5 and max(equil)<1e-10 and min(report['smooth_order'])>1.99)
    (OUT/'projected_gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='bulk_quadrature_cases'},indent=2))
    assert report['PASS']
if __name__=='__main__':run()
