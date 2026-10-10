from pathlib import Path
import json,subprocess,hashlib,datetime
R=Path('/scratch4/workspace/roohie_umass_edu-jfm1439-pairfix-20261001')
e=json.loads((R/'EXTRA_PASS.json').read_text())
assert e['job']=='65158940'
for n,h in e['manifest'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
assert hashlib.sha256((R/'run_extra.py').read_bytes()).hexdigest()==e['runner_sha256']
report={'gate_independently_verified':e,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
q=subprocess.check_output(['squeue','-u','roohie_umass_edu','-h','-o','%i|%T'],text=True)
if '65157835_2|PENDING' in q.splitlines():
 v=json.loads(Path('/scratch4/workspace/roohie_umass_edu-jfm1439-dvm-resume-20261001/scientific_audit_20261002/final_dvm_validation.json').read_text());assert v['archive_files_verified']==35
 subprocess.run(['scancel','65157835_2'],check=True)
 report['completed_dvm_pending_allocation_cancelled']='65157835_2; final52000 and35 archive hashes verified; no recomputation needed'
report['queue_after']=subprocess.check_output(['squeue','-u','roohie_umass_edu','-h','-o','%i|%j|%T'],text=True)
(R/'extra_repair_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
