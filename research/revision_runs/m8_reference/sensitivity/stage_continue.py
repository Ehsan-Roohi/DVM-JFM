import json,torch,hashlib,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;baseline=p.parent;out=p/'time_extend'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
a=json.loads((baseline/'config.json').read_text());b=json.loads((out/'config.json').read_text())
assert {k:v for k,v in a.items() if k!='max_time'}=={k:v for k,v in b.items() if k!='max_time'}
assert sha(baseline/'solver.py')==sha(out/'solver.py')
if (out/'checkpoint.pt').exists():
    assert (out/'continuation_provenance.json').exists();raise SystemExit('Already staged')
z=torch.load(baseline/'checkpoint.pt',weights_only=False,map_location='cpu')
assert z['signature']==sha(baseline/'solver.py')+sha(baseline/'config.json')
assert torch.isfinite(z['g']).all() and torch.isfinite(z['h']).all()
old=z['signature'];z['signature']=sha(out/'solver.py')+sha(out/'config.json')
torch.save(z,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
(out/'continuation_provenance.json').write_text(json.dumps({'reason':'Only maximum physical time extended400to800; physics/grid/stoppingcriteria unchanged','old_signature':old,'new_signature':z['signature'],'source_checkpoint_sha':sha(baseline/'checkpoint.pt'),'staged_checkpoint_sha':sha(out/'checkpoint.pt'),'step':z['step'],'old_config':a,'new_config':b},indent=2))
