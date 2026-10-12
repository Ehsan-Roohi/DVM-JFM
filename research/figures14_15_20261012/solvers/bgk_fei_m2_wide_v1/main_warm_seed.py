from pathlib import Path
import json,hashlib,time
import numpy as np
import bgk as new,bgk_reference as old,shakhov_reference as ref
import resource_watchdog as resource
assert resource.memory()['available_bytes']>=4*1024**3
P=Path(__file__).parent;src=P.parent/'bgk_fei_m2_native_v1/runs/M2_base'
s=json.loads((src/'STATUS.json').read_text());assert s['reason']=='signal_checkpoint'
for name,h in s['sha256'].items():assert ref.sha(src/name)==h
oc=s['config'];c=json.loads((P/'manifest.json').read_text())['cases'][0]
z=np.load(src/'checkpoint.npz',allow_pickle=False)
oldsig=hashlib.sha256((json.dumps(oc,sort_keys=True)+ref.sha(P/'bgk.py')+ref.sha(P/'shakhov_reference.py')).encode()).hexdigest()
assert str(z['signature'])==oldsig==s['signature']
x,v,w,f,l,r=ref.setup(c);ox,ov,ow,unused,ol,orr=ref.setup(oc)
assert np.array_equal(v,ov) and np.array_equal(w,ow)
offset=(c['nx']-oc['nx'])//2;assert np.array_equal(x[offset:offset+oc['nx']],ox)
f[:,:offset]=l[:,None];f[:,offset+oc['nx']:]=r[:,None];f[:,offset:offset+oc['nx']]=z['f']
a=f.copy();b=f.copy();dt=c['CFL']*2*c['L']/c['nx']/c['vmax']
for _ in range(20):a=old.advance(a,v,w,l,r,c,dt)
for _ in range(20):b=new.advance(b,v,w,l,r,c,dt)
fe=float(np.max(abs(a-b)/np.maximum(abs(a),1)));am=ref.moments(a,v,w);bm=ref.moments(b,v,w)
me=float(np.max(abs(am-bm)/np.maximum(abs(am),1)));assert fe<1e-11 and me<1e-10
dest=P/'runs/M2_domain';dest.mkdir(exist_ok=False)
sig=hashlib.sha256((json.dumps(c,sort_keys=True)+ref.sha(P/'bgk.py')+ref.sha(P/'shakhov_reference.py')).encode()).hexdigest()
ref.save(dest/'checkpoint.npz',f=f,step=0,stable=0,prev=np.array([]),signature=sig)
(P/'MAIN_SEED_ORACLE.json').write_text(json.dumps({'PASS':True,'source_checkpoint':ref.sha(src/'checkpoint.npz'),'initial_checkpoint':ref.sha(dest/'checkpoint.npz'),'distribution_error20':fe,'moment_error20':me,'geometry_cells_copied':oc['nx'],'reservoir_cells_added':c['nx']-oc['nx'],'step_reset':True,'window_reset':True,'probe_steps_not_saved':True,'source':ref.sha(__file__)},indent=2));print('MAIN_WARM_SEED_PASS',fe,me)
