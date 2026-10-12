from pathlib import Path
import json,time,math,sys
P=Path(__file__).parent
names=['M2_domain','M2_space','M2_velocity','M2_halfdt','M2_domain90'];previous=None
for _ in range(120):
    rows=[];fingerprint=[]
    for name in names:
        d=P/'runs'/name
        q=d/'STATUS.json'
        if q.exists():
            r=json.loads(q.read_text());rows.append({'case':name,'closed':True,'t':r['time'],'rhs':r['final_rhs'],'stationary':r['stationary']});fingerprint.append(('closed',r['stationary']))
        elif (d/'PROGRESS.json').exists():
            r=json.loads((d/'PROGRESS.json').read_text());rows.append({'case':name,'t':r['t'],'rhs':r['rhs'],'stable':r['stable']});fingerprint.append(('running',math.floor(math.log10(max(r['rhs'],1e-30))),bool(r['stable'])))
        else:rows.append({'case':name,'state':'QUEUED_OR_INITIALIZING'});fingerprint.append('queued')
    if fingerprint!=previous:print(json.dumps(rows),flush=True);previous=fingerprint
    if (P/'COMPARISON.json').exists():print('COMPARISON_READY',flush=True);sys.exit(0)
    if (P/'COMPARISON_NOT_QUALIFIED.json').exists():print('COMPARISON_NOT_QUALIFIED',flush=True);sys.exit(2)
    time.sleep(60)
sys.exit(3)
