from pathlib import Path
import hashlib,json,time
import numpy as np
import bgk as new,bgk_reference as old,shakhov_reference as ref
P=Path(__file__).parent;O=P.parent/'bgk_fei_m2_native_v1'
oldcases=json.loads((O/'manifest.json').read_text())['cases'];cases=json.loads((P/'manifest.json').read_text())['cases']
records={}
for ni,oi in [(1,1),(2,2),(3,3),(4,0)]:
    c=cases[ni];oc=oldcases[oi];src=O/'runs'/oc['name'];s=json.loads((src/'STATUS.json').read_text())
    assert s['config']==oc and s['reason']=='signal_checkpoint'
    for n,h in s['sha256'].items():assert ref.sha(src/n)==h
    z=np.load(src/'checkpoint.npz',allow_pickle=False);oldf=z['f']
    osig=hashlib.sha256((json.dumps(oc,sort_keys=True)+ref.sha(O/'bgk.py')+ref.sha(O/'shakhov_reference.py')).encode()).hexdigest()
    assert str(z['signature'])==osig==s['signature']
    x,v,w,f,l,r=ref.setup(c);ox,ov,ow,unused,ol,orr=ref.setup(oc)
    assert np.array_equal(v,ov) and np.array_equal(w,ow)
    offset=(c['nx']-oc['nx'])//2
    assert np.array_equal(x[offset:offset+oc['nx']],ox)
    f[:,:offset,:]=l[:,None,:];f[:,offset+oc['nx']:,:]=r[:,None,:]
    f[:,offset:offset+oc['nx'],:]=oldf
    assert np.array_equal(f[:,offset:offset+oc['nx']],oldf)
    # Changed-domain field is an initial guess; no original stationarity credit.
    dt=c['CFL']*2*c['L']/c['nx']/c['vmax'];a=f.copy();b=f.copy()
    start=time.monotonic()
    for _ in range(20):a=old.advance(a,v,w,l,r,c,dt)
    ot=time.monotonic()-start;start=time.monotonic()
    for _ in range(20):b=new.advance(b,v,w,l,r,c,dt)
    nt=time.monotonic()-start
    fe=float(np.max(abs(a-b)/np.maximum(abs(a),1)))
    am=ref.moments(a,v,w);bm=ref.moments(b,v,w)
    me=float(np.max(abs(am-bm)/np.maximum(abs(am),1)))
    assert fe<1e-11 and me<1e-10
    out=P/'runs'/c['name'];out.mkdir(parents=True,exist_ok=False)
    sig=hashlib.sha256((json.dumps(c,sort_keys=True)+ref.sha(P/'bgk.py')+ref.sha(P/'shakhov_reference.py')).encode()).hexdigest()
    ref.save(out/'checkpoint.npz',f=f,step=0,stable=0,prev=np.array([]),signature=sig)
    records[c['name']]={'input_checkpoint':ref.sha(src/'checkpoint.npz'),'output_checkpoint':ref.sha(out/'checkpoint.npz'),'original_physical_time':s['time'],'physical_time_reset_for_changed_domain':True,'exact_cells_preserved':oc['nx'],'reservoir_cells_appended':c['nx']-oc['nx'],'distribution_error20':fe,'moments_error20':me,'speedup20':ot/nt,'no_probe_step_saved':True}
q=json.loads((P/'NATIVE_GATE.json').read_text());q['extended_saved_state_oracles']=records;q['scope']='Cold and exact-cell reservoir-extended saved-state oracles; physical geometry changed prospectively, convergence credit reset';q['bindings']['bridge.py']=ref.sha(__file__)
(P/'NATIVE_GATE.json').write_text(json.dumps(q,indent=2));(P/'BRIDGE_RECEIPT.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
