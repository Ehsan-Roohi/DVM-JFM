import ast,json,subprocess,sys,hashlib,os
from pathlib import Path
R=Path(__file__).resolve().parent
source=ast.parse((R/'run.py').read_text());source.body=[n for n in source.body if not isinstance(n,ast.If)]
ns={'__file__':str(R/'run.py')};exec(compile(source,'run.py','exec'),ns)
assert ns['manifest']()==json.loads((R/'PASS.json').read_text())['manifest']
d=R/('extra_gate_'+os.environ['SLURM_JOB_ID']);d.mkdir(exist_ok=True)
subprocess.run(['g++','-O3','-std=c++17',str(R/'collision_test.cpp'),'-o',str(d/'collision_test')],check=True)
subprocess.run([str(d/'collision_test')],check=True)
for ppc in [120,960]:
 for mode in ['original','corrected']:
  for label,steps in [('continuous',100),('split',40),('split',100)]:
   subprocess.run(ns['cmd'](mode,d/f'{mode}_{ppc}_{label}',ppc,99123,steps,20,2,20),check=True,stdout=subprocess.DEVNULL)
  assert ns['validate'](d/f'{mode}_{ppc}_continuous',mode,100,41)==ns['validate'](d/f'{mode}_{ppc}_split',mode,100,41)
proof={'manifest':ns['manifest'](),'runner_sha256':ns['sha'](R/'run_extra.py'),'gate_script_sha256':ns['sha'](Path(__file__)),'job':os.environ['SLURM_JOB_ID'],'checks':'fresh continuous/split parity at120/960 both versions, conservation10000trials'}
(R/'EXTRA_PASS.json').write_text(json.dumps(proof,indent=2));print('EXTRA GATE PASSED',flush=True)
