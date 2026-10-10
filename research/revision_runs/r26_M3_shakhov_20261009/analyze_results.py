from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'stf_repaired_tensor'))
import shock_pilot as tensor_maxwell
import r26_shakhov_tensor as tensor_shakhov
sys.path.insert(0,str(HERE/'physics'))
import maxwell_repaired_mixed as scalar_maxwell
import projected_mixed as scalar_shakhov
FIELDS=['rho','u','theta','sigma','q','m','R','Delta','S']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def midpoint(x,rho):
    ids=np.flatnonzero((rho[:-1]-2)*(rho[1:]-2)<=0);i=ids[np.argmin(abs(x[ids]))]
    return x[i]+(2-rho[i])*(x[i+1]-x[i])/(rho[i+1]-rho[i])
def features(x,rho,T):
    xx=x-midpoint(x,rho);threshold=1+.01*(11/3-1)
    ii=np.flatnonzero((T[:-1]-threshold)*(T[1:]-threshold)<=0);i=ii[0]
    onset=xx[i]+(threshold-T[i])*(xx[i+1]-xx[i])/(T[i+1]-T[i])
    pk=np.argmax(T)
    return dict(T_peak=float(T[pk]),T_peak_x=float(xx[pk]),T_overshoot=float(T[pk]-11/3),T_overshoot_pct_downstream=float(100*(T[pk]-11/3)/(11/3)),T_1pct_onset=float(onset),rho_min=float(min(rho)),rho_max=float(max(rho)),rho_negative_steps=int(np.sum(np.diff(rho)<-1e-10)))
report={'cases':{},'scientific_validation':False,'limitations':'Independent numerical/source gates do not certify physical solution error. Shakhov collision model differs from Maxwell Boltzmann. DVM reference is nonstationary; no fit/smoothing/width rescaling.'}
for model,prefix,nodes in [('maxwell','maxwell_stf',[961,1921,3841,7681]),('shakhov','projected',[961,1921,3841,7681,15361])]:
    tm=tensor_maxwell if model=='maxwell' else tensor_shakhov
    sm=scalar_maxwell if model=='maxwell' else scalar_shakhov
    for n in nodes:
        case=prefix+str(n);folder=HERE/case;a=np.load(folder/'profile.npz',allow_pickle=False);p=tm.Shock(n,45,3)
        t,r,eq=p.evaluate(a['z']);f=p.residual(a['z']);expected={'rho':t.rho,'u':t.velocity[:,0],'theta':t.theta,'sigma':t.sigma[:,0,0],'q':t.heat_flux[:,0],'m':t.m[:,0,0,0],'R':t.R[:,0,0],'Delta':t.Delta,'S':t.R[:,0,0]+t.Delta/3}
        reconstruction=max(float(np.max(abs(a[k]-expected[k]))) for k in FIELDS)
        finite=all(np.isfinite(a[k]).all() for k in a.files)
        config=json.loads((folder/'manifest.json').read_text())['config']
        actual_hash_match=all(sha(HERE/k)==v for k,v in config['source_hashes'].items())
        pressure=min(np.min(t.rho*t.theta+t.sigma[:,0,0]),np.min(t.rho*t.theta-t.sigma[:,0,0]/2))
        pde=float(np.max(abs(eq/p.eqscale)));bc=float(max(abs(f[-11:])));scalar=sm.nodal_rows(p,a['Y'])
        own_vs_tensor=float(np.max(abs(scalar-eq)/p.eqscale))
        rec=dict(model=model,N=n,L=45,source_hashes_match=actual_hash_match,coordinates_match=bool(np.array_equal(a['x'],p.x)),all_finite=finite,nine_field_reconstruction=reconstruction,independent_tensor_PDE=pde,independent_boundary_phase=bc,minpressure=float(pressure),scalar_tensor_difference=own_vs_tensor,profile_sha=sha(folder/'profile.npz'),**features(a['x'],a['rho'],a['theta']))
        rec['independent_numerical_pass']=bool(pde<.01 and bc<1e-4 and pressure>0 and finite and reconstruction==0 and actual_hash_match and rec['coordinates_match'] and own_vs_tensor<1e-9)
        report['cases'][case]=rec
    # Difference order on common nodes; no certified error estimate.
    diffs=[]
    for n1,n2 in zip(nodes[:-1],nodes[1:]):
        a=np.load(HERE/(prefix+str(n1))/'profile.npz');b=np.load(HERE/(prefix+str(n2))/'profile.npz')
        diffs.append({k:float(np.linalg.norm(a[k]-b[k][::2])/np.linalg.norm(b[k][::2])) for k in FIELDS})
    report[model+'_grid_relative_L2']=diffs
    report[model+'_observed_order']=[{k:float(np.log2(diffs[i][k]/diffs[i+1][k])) for k in FIELDS} for i in range(len(diffs)-1)]
oldpath=HERE.parent/'r26_M3_discretization_repair_20261008/mixed7681/profile.npz';old=np.load(oldpath)
assert sha(oldpath)=='41b9c7b8e56b3429a010d60f3d44ca0acc641cc4d8de4148f3f5cfe16898a902'
report['old_Maxwell_preserved_sha']=sha(oldpath);report['old_Maxwell_shape']=features(old['x'],old['rho'],old['theta'])
curves=[('Previous R26 Maxwell',old,'#d62728','-'),('STF-corrected R26 Maxwell',np.load(HERE/'maxwell_stf7681/profile.npz'),'#16823b','-'),('R26 Shakhov',np.load(HERE/'projected15361/profile.npz'),'#6c42c1','-')]
dv=np.load(HERE.parent/'r26_M3_matched_dvm_diagnostic_20261008/profiles.npz');dfields={str(k):dv['fields'][:,i] for i,k in enumerate(dv['columns'])};dfields['theta']=dfields['T'];dfields['x']=dv['x']
curves.append(('Shakhov DVM snapshot (not stationary)',dfields,'#222222','--'))
fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for name,a,color,ls in curves:
    x=a['x']-midpoint(a['x'],a['rho']);T=a['theta'];rho=a['rho']
    ax[0,0].plot(x,T,c=color,ls=ls,label=name);ax[0,1].plot(x,rho,c=color,ls=ls)
    ax[1,0].plot(x,np.gradient(T,x,edge_order=2),c=color,ls=ls);ax[1,1].plot(x,np.gradient(rho,x,edge_order=2),c=color,ls=ls)
for a,ylabel in zip(ax.flat,['Temperature T/T1','Density rho/rho1','dT/dx (no smoothing)','drho/dx (no smoothing)']):
    a.set(xlim=(-18,12),xlabel='x - density midpoint [mu1/(rho1 sqrt(T1))]',ylabel=ylabel);a.grid(alpha=.2)
ax[0,0].legend(fontsize=8);fig.suptitle('Mach 3: printed STF bracket corrections and isolated Shakhov R26\nSame Pr=2/3, mu=T; numerical gates passed on each displayed R26 variant; no physical-validation claim')
for ext in ['png','pdf','svg']:fig.savefig(HERE/('M3_STF_and_Shakhov.'+ext),dpi=170)
(HERE/'independent_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'old':report['old_Maxwell_shape'],'corrected_Maxwell':report['cases']['maxwell_stf7681'],'Shakhov':report['cases']['projected15361']},indent=2))
