from pathlib import Path
import json, hashlib, datetime, importlib.util, torch, numpy as np, zipfile, struct
D=Path('/scratch4/workspace/roohie_umass_edu-jfm1439-dvm-resume-20261001/active')
d=D/'outputs/dvm/M5_velocity_refine'
out=Path(__file__).parent/'final_dvm_validation.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
m=json.loads((D/'migration.json').read_text());g=json.loads((D/'GATE32_PASS.json').read_text())
assert sha(D/'source_sha256.json')==m['source_manifest']==g['manifest']
assert sha(D/'cases.json')==m['cases_sha256']==g['cases']
assert sha(D/'run_dvm_resume.py')==g['runner']
for n,h in json.loads((D/'source_sha256.json').read_text()).items():assert sha(D/n)==h
c=json.loads((D/'cases.json').read_text())[2];done=json.loads((d/'DONE.json').read_text());settings=json.loads((d/'result.npz.settings.json').read_text())
for k,v in c.items():
 if k!='name':assert settings[k]==v
ck=torch.load(d/'restart.pt',map_location='cpu',weights_only=False,mmap=True);torch.set_num_threads(1)
assert ck['step']==c['steps']==52000
assert ck['signature']=={k:v for k,v in settings.items() if k not in ('out','fig','checkpoint','checkpoint_every','steps','device','save_every')}
for key in ['f','last_rho']:
 for start in range(0,len(ck[key]),16):assert torch.isfinite(ck[key][start:start+16]).all()
with np.load(d/'result.npz',allow_pickle=False) as z:
 for key in ['rho','ux','T','qx','sigma_xx','Rxx_closure','Delta_dvm','S_dvm']:assert z[key].size==c['nx'] and np.isfinite(z[key]).all()
a=json.loads((d/'ARCHIVED.json').read_text());dest=Path(a['archive']);assert json.loads((dest/'ARCHIVED.json').read_text())==a
for n,h in a['files'].items():
 assert sha(d/n)==h and sha(dest/n)==h,n
 print('verified',n,flush=True)
report={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':c,'checkpoint_step':52000,'finite_checkpoint_and_final':True,'source_gate_configuration_match':True,'archive_files_verified':len(a['files']),'archive':str(dest),'done':done}
out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
