"""Report available refinement comparisons without substituting for convergence."""
import json
from pathlib import Path
import numpy as np
import shakhov_reference as ref
ROOT=Path(__file__).resolve().parent
def compare(mach,first,second):
    a=ROOT/'runs'/f'M{mach}_{first}';b=ROOT/'runs'/f'M{mach}_{second}'
    if not (a/'STATUS.json').exists() or not (b/'STATUS.json').exists():return dict(available=False)
    sa=json.loads((a/'STATUS.json').read_text());sb=json.loads((b/'STATUS.json').read_text())
    za=np.load(a/'profiles.npz');zb=np.load(b/'profiles.npz');xa,aa=za['x'],za['fields'];xb,bb=zb['x'],zb['fields']
    aa=np.column_stack([np.interp(xb,xa,aa[:,i]) for i in range(8)])
    core=abs(xb)<.8*min(sa['config']['L'],sb['config']['L'])
    errors=np.linalg.norm((aa-bb)[core],axis=0)/np.maximum(np.linalg.norm(bb[core],axis=0),1e-12)
    comparable=(sa['stationary'] and sb['stationary']) or abs(sa['time']-sb['time'])<max(sa['dt'],sb['dt'])
    return dict(available=True,time_comparable=bool(comparable),both_stationary=bool(sa['stationary'] and sb['stationary']),times=[sa['time'],sb['time']],relative_L2=dict(zip(['rho','u','T','q','sigma','R','Delta','S'],map(float,errors))),comparison='same x coordinates; no translation fitted; interior abs(x)<0.8L',certified_order=False)
reports={}
for mach in [2,3]:
    reports[str(mach)]={label:compare(mach,a,b) for label,a,b in [('space','base','x2'),('velocity','x2','v2'),('time','x2','dt2')]}
    for kind in ['base','x2','v2','dt2']:
        path=ROOT/'runs'/f'M{mach}_{kind}'/'STATUS.json'
        if path.exists():
            s=json.loads(path.read_text());reports[str(mach)][kind]={k:s[k] for k in ['stationary','time','final_rhs','face_flux_relative_spans','budgets','reason']}
ref.dump(ROOT/'COMPARISON.json',dict(cases=reports,note='Available comparisons only. Reaching tmax does not certify stationarity or accuracy; no automatic scientific acceptance.'))
