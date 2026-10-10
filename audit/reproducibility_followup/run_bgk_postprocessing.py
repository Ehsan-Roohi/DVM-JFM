"""Parent-authorized three sequential archival reductions;600s per owned child."""
import json,os,subprocess,sys,time
from pathlib import Path
from resource_check import available
HERE=Path(__file__).resolve().parent
records=[]
for mach in (2,3,5):
    first=available()
    if first<4:
        records.append({'Mach':mach,'status':'HELD_INITIAL_RAM','available_GiB':first});break
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    start=time.monotonic();failure=None
    with (HERE/f'BGK_M{mach}_stdout.log').open('w') as out,(HERE/f'BGK_M{mach}_stderr.log').open('w') as err:
        child=subprocess.Popen([sys.executable,str(HERE/'regenerate_bgk_budgets.py'),'--mach',str(mach),'--chunk','4'],env=env,stdout=out,stderr=err)
        while child.poll() is None:
            if time.monotonic()-start>600:
                failure='EXTERNAL_600s_LIMIT';child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(1)
    records.append({'Mach':mach,'exit_code':child.returncode,'failure':failure,'elapsed_s':time.monotonic()-start,'initial_available_GiB':first,'kind':'POSTPROCESS_ONLY'})
    (HERE/'BGK_POSTPROCESS_EXECUTION_RECEIPT.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records[-1]),flush=True)
    if failure or child.returncode:break
if len(records)!=3 or any(r.get('exit_code')!=0 for r in records):sys.exit(1)
