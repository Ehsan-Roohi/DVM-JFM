from pathlib import Path
import json,shutil,hashlib
P=Path(__file__).parent;src=P.parent/'bgk_fei_m2_native_v1/runs/M2_domain'
dest=P/'runs/M2_domain'
if dest.exists():
    # A separately reviewed warm baseline was launched here; never overwrite it.
    assert (P/'MAIN_SEED_ORACLE.json').exists() and (dest/'STATUS.json').exists()
    print('Using terminal reviewed warm baseline');raise SystemExit(0)
s=json.loads((src/'STATUS.json').read_text());c=json.loads((P/'manifest.json').read_text())['cases'][0]
assert s['config']==c
for name,h in s['sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==h
assert not dest.exists()
shutil.copytree(src,dest)
(P/'BASELINE_COPY_RECEIPT.json').write_text(json.dumps({'source':str(src),'source_closed':True,'stationary':s['stationary'],'sha256':s['sha256'],'no_acceptance_assumed':True},indent=2))
