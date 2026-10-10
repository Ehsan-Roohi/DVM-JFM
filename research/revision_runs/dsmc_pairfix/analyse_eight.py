from pathlib import Path
import zipfile,json,hashlib,numpy as np
r=Path('revision_runs/dsmc_pairfix');z=zipfile.ZipFile(r/'corrected_final_csvs.zip');keys=['Rxx_closure','Delta','S','QDelta_collision'];out={'method':'Eight seeds; linear density midpoint alignment on |x|<=15. Paired-label SE is descriptive; RNG trajectories diverge. Bootstrap norm intervals do not test zero effect. Corrected algorithm separate from original convergence campaign. Archive revalidation currently blocked by project availability.','results':{},'sha256':{}}
def align(a,k):
 x=a['x'];n=a['n'];ii=np.where((n[:-1]<2)&(n[1:]>=2))[0];assert len(ii)==1;i=ii[0];c=x[i]+(2-n[i])*(x[i+1]-x[i])/(n[i+1]-n[i]);g=x[abs(x)<=15];return np.interp(g+c,x,a[k])
rng=np.random.default_rng(20261002)
for ppc in [120,960]:
 aa=[];bb=[]
 for seed in range(1,9):
  p=Path(f'revision_runs/completed_analysis/ppc_convergence_3840/inputs/ppc{ppc}_seed{seed}/M3_final.csv');raw=p.read_bytes();rawb=z.read(f'corrected_ppc{ppc}_seed{seed}/M3_final.csv')
  out['sha256'][str(p)]=hashlib.sha256(raw).hexdigest();out['sha256'][f'corrected_{ppc}_{seed}']=hashlib.sha256(rawb).hexdigest()
  import io
  a=np.genfromtxt(io.BytesIO(raw),delimiter=',',names=True);b=np.genfromtxt(io.BytesIO(rawb),delimiter=',',names=True)
  assert all(np.isfinite(t[k]).all() for t in [a,b] for k in t.dtype.names)
  aa.append({k:align(a,k) for k in keys});bb.append({k:align(b,k) for k in keys})
 vals={}
 for k in keys:
  a=np.array([i[k] for i in aa]);b=np.array([i[k] for i in bb]);d=b-a;den=np.linalg.norm(a.mean(0));ix=rng.integers(0,8,(2000,8));boot=np.array([100*np.linalg.norm(d[j].mean(0))/np.linalg.norm(a[j].mean(0)) for j in ix])
  vals[k]={'mean_difference_L2_percent':float(100*np.linalg.norm(d.mean(0))/den),'paired_SE_norm_percent':float(100*np.linalg.norm(d.std(0,ddof=1)/np.sqrt(8))/den),'descriptive_bootstrap_norm95':np.quantile(boot,[.025,.975]).tolist()}
 out['results'][str(ppc)]=vals
(r/'eight_seed_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['results']))
