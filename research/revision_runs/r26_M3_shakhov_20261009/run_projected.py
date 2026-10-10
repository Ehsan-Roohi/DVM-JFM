"""Controlled isolated Shakhov26 solve; no target fit or automatic successor."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import bmat,eye,diags
from scipy.sparse.linalg import splu
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'physics'));import projected_mixed as im
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def checkpoint(path,**data):
    tmp=path.with_suffix('.tmp')
    with open(tmp,'wb') as f:np.savez(f,**data)
    os.replace(tmp,path)

def run(args):
    assert json.loads((HERE/'projected_gate.json').read_text())['PASS']
    global im
    if args.model=='maxwell':
        assert json.loads((HERE/'maxwell_repair_gate.json').read_text())['PASS']
        import maxwell_repaired_mixed as im
    OUT=HERE/args.out;OUT.mkdir(exist_ok=False)
    p=im.Shock(args.nodes,args.domain,3);a=np.load(args.seed,allow_pickle=False)
    zz=np.column_stack([np.interp(p.x,a['x'],a['z'][:,k]) for k in range(5)])
    # For an expanded domain impose RH equilibrium outside seed extent.
    zz[p.x<a['x'][0]]=p.left/p.scale;zz[p.x>a['x'][-1]]=p.right/p.scale
    Y=im.lift(p,zz);z=Y.ravel();pattern=im.sparsity(p)
    fun=lambda z:im.residual_diagnostic(p,z)
    def metrics(z):
        Y=z.reshape(p.n,8);f=fun(z);t=p.fields(Y[:,:5]);nodal=im.nodal_rows(p,Y)/p.eqscale
        return {'midpoint_max':float(np.max(abs(f[:-11]))),'own_full_nodal_PDE':float(np.max(abs(nodal))),'boundary_phase':float(np.max(abs(f[-11:]))/100),'minpressure':float(min(np.min(t.rho*t.theta+t.sigma[:,0,0]),np.min(t.rho*t.theta-t.sigma[:,0,0]/2))),'closure_gap':float(np.max(abs(im.point_rows(p,Y,np.gradient(Y,p.x,axis=0,edge_order=2))[:,2:5])))}
    J=approx_derivative(fun,z,method='3-point',abs_step=1e-5,sparsity=pattern).tocsr()
    v=np.random.default_rng(123).normal(size=z.size);v/=np.linalg.norm(v);fd=(fun(z+1e-5*v)-fun(z-1e-5*v))/2e-5
    jv=float(np.linalg.norm(J@v-fd)/np.linalg.norm(fd));assert jv<1e-5
    hashes={str(f.relative_to(HERE)):sha(f) for f in (HERE/'physics').rglob('*.py')}
    config={'M':3,'N':args.nodes,'L':args.domain,'maxiterations':args.maxiterations,'Pr':2/3,'mu':'T','tau':'1/rho','closure':'first-order kinetic projection on Grad26' if args.model=='shakhov' else 'printed Gu nonlinear closures with three STF bracket repairs','collision':'Shakhov' if args.model=='shakhov' else 'Maxwell molecules','endpoint_and_phase_weight':100,'equation_tolerance_strict':.01,'boundary_phase_tolerance_strict':1e-4,'source_hashes':hashes}
    dump(OUT/'manifest.json',dict(config=config,seed=str(Path(args.seed).resolve()),seed_sha=sha(Path(args.seed)),actual_gate_Jv=jv,rows=8*(p.n-1)+11,unknowns=z.size,automatic_acceptance=False,scientific_validation=False))
    f=fun(z);lam=1e-6;hist=[];start=time.time();reason='ITERATION_LIMIT'
    checkpoint(OUT/'initial.npz',x=p.x,Y=Y,z=Y[:,:5])
    for it in range(args.maxiterations+1):
        m=metrics(z);rec=dict(iteration=it,cost=float(f@f/2),seconds=time.time()-start,**m);hist.append(rec)
        dump(OUT/'history.json',hist);print(json.dumps(rec),flush=True)
        checkpoint(OUT/'checkpoint.npz',x=p.x,Y=z.reshape(p.n,8),z=z.reshape(p.n,8)[:,:5],lam=lam,iteration=it,config=json.dumps(config))
        if it==args.maxiterations:break
        if it and len(hist)>5 and abs(hist[-1]['cost']-hist[-6]['cost'])/max(hist[-6]['cost'],1e-30)<1e-9:reason='PLATEAU';break
        if it:J=approx_derivative(fun,z,method='3-point',abs_step=1e-5,sparsity=pattern).tocsr()
        scale=1/np.maximum(np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel()),1e-8);A=J@diags(scale);ok=False
        for retry in range(10):
            K=bmat([[eye(A.shape[0]),A],[A.T,-lam*eye(A.shape[1])]],format='csc');step=scale*splu(K).solve(np.r_[-f,np.zeros(z.size)])[A.shape[0]:]
            for alpha in [1,.5,.25,.125,.0625,.03125]:
                zz=z+alpha*step
                try:
                    ff=fun(zz);mm=metrics(zz)
                    if np.isfinite(ff).all() and mm['minpressure']>0 and ff@ff<f@f:z=zz;f=ff;lam=max(lam/3,1e-14);ok=True;break
                except (ValueError,FloatingPointError):pass
            if ok:break
            lam*=10
        if not ok:reason='NO_DESCENT';break
    Y=z.reshape(p.n,8);t=p.fields(Y[:,:5]);m=metrics(z)
    checkpoint(OUT/'profile.npz',x=p.x,Y=Y,z=Y[:,:5],rho=t.rho,u=t.velocity[:,0],theta=t.theta,sigma=t.sigma[:,0,0],q=t.heat_flux[:,0],m=t.m[:,0,0,0],R=t.R[:,0,0],Delta=t.Delta,S=t.R[:,0,0]+t.Delta/3)
    report=dict(**m,reason=reason,iterations=it,profile_sha=sha(OUT/'profile.npz'),own_numerical_criteria_pass=bool(m['own_full_nodal_PDE']<.01 and m['boundary_phase']<1e-4 and m['minpressure']>0),scientific_validation=False)
    dump(OUT/'RESULT.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--nodes',type=int,required=True);ap.add_argument('--domain',type=float,default=45);ap.add_argument('--seed',required=True);ap.add_argument('--out',required=True);ap.add_argument('--maxiterations',type=int,default=20);ap.add_argument('--model',choices=['shakhov','maxwell'],default='shakhov')
    run(ap.parse_args())
