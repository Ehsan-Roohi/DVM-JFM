from pathlib import Path
import hashlib,json,subprocess,shutil
import numpy as np
import postprocess as pp
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((R/'manifest.json').read_text())
for name,h in m['sources'].items():assert sha(R/name)==h,name
results={'moment_tests':pp.selftest(),'cases':[]}
def run(d,resume):
 with (d/('resume.log' if resume else 'initial.log')).open('w') as log:
  subprocess.run([str(R/'bird_shock')],cwd=d,input='0\n0\n' if resume else '1\n1\n',text=True,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
for c in m['cases']:
 d=R/'gate_runs_v4'/('M'+str(c['Mach']));assert not d.exists();(d/'blocks').mkdir(parents=True)
 (d/'case.in').write_text(f"{c['Mach']} {c['seed']} 10 .005 0 2 2\n")
 run(d,False);h,a=pp.load_block(d/'blocks/block_000002.dat');f=pp.fields(a[:,1:57],a[:,57:92]);assert np.min(f['rho'])>0 and np.min(f['theta'])>0
 q=a[:,57:92];conserv=max(np.max(abs(q[:,0])),*(np.max(abs(q[:,pp.IDX[t]])) for t in [(1,0,0),(0,1,0),(0,0,1)]),np.max(abs(q[:,pp.IDX[2,0,0]]+q[:,pp.IDX[0,2,0]]+q[:,pp.IDX[0,0,2]])))
 assert conserv<1e-8,conserv
 results['cases'].append({'Mach':c['Mach'],'finite_positive':True,'collision_conservation_max':float(conserv),'checkpoint_sha':sha(d/'DSMC1S.RES')})
 if c['Mach']==2:
  e=R/'gate_runs_v4'/'restart_parity';(e/'blocks').mkdir(parents=True);(e/'case.in').write_text(f"2 {c['seed']} 10 .005 0 2 1\n")
  run(e,False);assert not (e/'FINAL_TIME_REACHED').exists();run(e,True)
  assert sha(e/'blocks/block_000002.dat')==sha(d/'blocks/block_000002.dat'),'Restart trajectory mismatch'
  results['restart_block_bitwise']=True
results.update(PASS=True,manifest=sha(R/'manifest.json'),sources=m['sources'],executable=sha(R/'bird_shock'))
# A distinct two-block production-density smoke checks the raised particle
# capacity and native fourth-moment output before the long run is released.
c=m['cases'][0]
d=R/'gate_runs_v4'/'M2_high_ppc_smoke';assert not d.exists();(d/'blocks').mkdir(parents=True)
(d/'case.in').write_text(f"2 {c['seed']} {c['upstream_ppc']} {c['dt']} 0 2 2\n")
run(d,False)
h,a=pp.load_block(d/'blocks/block_000002.dat')
f=pp.fields(a[:,1:57],a[:,57:92])
assert np.isfinite(a).all() and np.min(f['rho'])>0 and np.min(f['theta'])>0
results['high_ppc_smoke']=dict(ppc=c['upstream_ppc'],final_block=2,finite_positive=True,
                               checkpoint_sha=sha(d/'DSMC1S.RES'),
                               max_abs_Delta=float(np.max(abs(f['Delta']))))
(R/'PASS.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
