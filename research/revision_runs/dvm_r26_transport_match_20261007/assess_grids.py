"""Three-grid differences on common physical coordinates, without width fitting.

Run on Unity after outputs exist. Reports pending inputs rather than declaring
grid independence. Observed order is descriptive, not an error bound.
"""
from pathlib import Path
import json,numpy as np
BASE=Path('/scratch3/workspace/roohie_umass_edu-jfm1439-r26-m2-20261007')
OUT=BASE/'dvm_transport_match/grid_assessment';OUT.mkdir(exist_ok=True)
fields=['rho','u','T','q','sigma','R','Delta','S']
groups={
 'R26':[
 BASE/'sweep_20261007/seed/profile.npz',
 BASE/'sweep_20261007/runs/M2_grid_N321/M2_N321_L30/profile.npz',
 BASE/'r26_grid641/runs/M2_grid_N641/M2_N641_L30/profile.npz'],
 'DVM_Shakhov':[BASE/'dvm_transport_match/runs'/name/'profiles.npz' for name in ['M2_base','M2_x640','M2_x1280']]}
result={'coordinate':'physical R26 length unit, density midpoint translation only; no width fitting','groups':{}}
for name,paths in groups.items():
    missing=[str(p) for p in paths if not p.exists()]
    if missing:
        result['groups'][name]={'state':'pending','missing':missing};continue
    profiles=[];eligible=True;checks=[]
    for p in paths:
        a=np.load(p,allow_pickle=False);x=a['x']
        if name=='R26':
            f=np.column_stack([a['theta'] if k=='T' else a[k] for k in fields])
            cp=p.parent/'INDEPENDENT_CHECK.json'
            if cp.exists():check=json.loads(cp.read_text());ok=check['passed']
            else:check=json.loads((p.parent/'RESULT.json').read_text());ok=check['numerical_gate_pass']
        else:
            f=a['fields'];cp=p.parent/'DONE.json'
            check=json.loads(cp.read_text()) if cp.exists() else {};ok=check.get('stationary',False)
        assert np.isfinite(f).all()
        mid=(1+16/7)/2;ii=np.flatnonzero((f[:-1,0]-mid)*(f[1:,0]-mid)<=0)
        assert len(ii)==1,'Ambiguous density-midpoint alignment'
        i=ii[0];xs=x[i]+(mid-f[i,0])*(x[i+1]-x[i])/(f[i+1,0]-f[i,0])
        profiles.append((x-xs,f));eligible=eligible and ok;checks.append(check)
    xx=np.linspace(-15,15,1201)
    assert all(x[0]<=xx[0] and x[-1]>=xx[-1] for x,f in profiles)
    yy=[np.column_stack([np.interp(xx,x,f[:,i]) for i in range(8)]) for x,f in profiles]
    den=np.linalg.norm(yy[-1],axis=0);e01=np.linalg.norm(yy[0]-yy[1],axis=0);e12=np.linalg.norm(yy[1]-yy[2],axis=0)
    values={}
    for i,k in enumerate(fields):
        values[k]={'coarse_medium_percent':float(100*e01[i]/max(den[i],1e-12)),
                   'medium_fine_percent':float(100*e12[i]/max(den[i],1e-12)),
                   'observed_order':float(np.log2(e01[i]/e12[i])) if e01[i]>1e-12 and e12[i]>1e-12 else None}
    result['groups'][name]={'state':'differences_computed','all_numerical_or_stationary_checks_pass':bool(eligible),'metrics':values,
                           'grid_independence_claim':False,'note':'Interpret decreasing differences with domain, velocity and time-step tests; no error tolerance chosen automatically.'}
(OUT/'three_grid_report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
