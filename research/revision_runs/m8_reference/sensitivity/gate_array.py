import json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
c=json.loads((p/'manifest.json').read_text())[int(os.environ['SLURM_ARRAY_TASK_ID'])]
if c['name']=='time_extend':
 subprocess.run([sys.executable,str(p/'stage_continue.py')],check=True)
for mode in ['cpu-gate','gpu-gate']:
 subprocess.run([sys.executable,'-u',str(Path(c['scratch'])/'solver.py'),mode],check=True,cwd=c['scratch'])
