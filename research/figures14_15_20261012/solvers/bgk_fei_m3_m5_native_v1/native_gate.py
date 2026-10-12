from pathlib import Path
import json,time,hashlib
import numpy as np
import bgk_reference as old
import bgk as new
import shakhov_reference as ref
P=Path(__file__).parent
report={}
for c in json.loads((P/'manifest.json').read_text())['cases']:
    x,v,w,f,l,r=ref.setup(c);dt=c['CFL']*2*c['L']/c['nx']/c['vmax']
    a=f.copy();b=f.copy();start=time.monotonic()
    for i in range(20):a=old.advance(a,v,w,l,r,c,dt)
    oldtime=time.monotonic()-start;start=time.monotonic()
    for i in range(20):b=new.advance(b,v,w,l,r,c,dt)
    newtime=time.monotonic()-start
    err=float(np.max(abs(a-b)/np.maximum(abs(a),1)))
    ma=ref.moments(a,v,w);mb=ref.moments(b,v,w)
    me=float(np.max(abs(ma-mb)/np.maximum(abs(ma),1)))
    assert err<1e-11 and me<1e-10,(c,err,me)
    report[c['name']]={'distribution_scaled_max':err,'eight_moment_scaled_max':me,'old_seconds_20':oldtime,'native_seconds_20':newtime,'speedup':oldtime/newtime}
files=['bgk.py','bgk_reference.py','kernel.c','kernel.so','native_kernel.py','native_gate.py','manifest.json','shakhov_reference.py']
r=dict(PASS=True,cold_twenty_step_oracles=report,bindings={n:ref.sha(P/n) for n in files},scope='same finite-grid upwind BGK map; cold mixtures only. Saved checkpoint oracle required before adoption.')
(P/'NATIVE_GATE.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
