import json, subprocess, hashlib, sys, datetime, os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SSH=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=6','-o','StrictHostKeyChecking=yes','-o','MACs=hmac-sha2-256-etm@openssh.com','-c','aes256-ctr','-i',str(Path.home()/'.ssh/unity_ed25519')]
HOST='roohie_umass_edu@login.unityhpc.org'
os.environ['PYTHONIOENCODING']='utf-8'
manifest=json.loads((ROOT/'extraction_manifest.json').read_text())
state={'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'device':'cpu','threads':2,'cases':{}}
def save():
    state['updated']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp=ROOT/'local_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(ROOT/'local_status.json')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
try:
    for i,c in enumerate(manifest['cases']):
        tag=c['tag'];d=ROOT/tag;d.mkdir(exist_ok=True)
        record=state['cases'][tag]={'stage':'transfer','files':{}};save()
        for key in ['profile','distribution']:
            remote=c['remote_'+key];target=Path(c[key])
            expected=subprocess.check_output(['ssh',*SSH,HOST,'sha256sum '+remote],text=True).split()[0]
            assert len(expected)==64
            if key=='profile':assert expected==c['profile_sha256']
            if not target.exists() or sha(target)!=expected:
                part=target.with_suffix('.downloading')
                subprocess.run(['scp',*SSH,HOST+':'+remote,str(part)],check=True)
                assert sha(part)==expected,'Transfer hash mismatch'
                part.replace(target)
            record['files'][key]={'path':str(target),'sha256':expected,'bytes':target.stat().st_size};save()
        record['stage']='cpu_analysis';save()
        with open(d/'run.log','a',encoding='utf-8') as log:
            subprocess.run([str(Path.home()/'AppData/Local/Temp/jfm_cpu/Scripts/python.exe'),'-u',str(ROOT/'extract_moments.py'),str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
        done=json.loads((d/'DONE.json').read_text())
        for n,h in done['files'].items():assert sha(d/n)==h
        record['stage']='completed';record['done']=done;save()
    subprocess.run([sys.executable,str(ROOT/'analyse_results.py')],check=True)
    state['complete']=True;save()
except Exception as exc:
    state['error']=repr(exc);save();raise
