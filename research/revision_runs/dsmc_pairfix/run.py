import os,sys,json,hashlib,subprocess,shutil,time,concurrent.futures,struct
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
A=Path(json.loads((R/'storage.json').read_text())['archive'])
assert str(R).startswith('/scratch')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():return {n:sha(R/n) for n in ['original.cpp','corrected.cpp','original','corrected','collision_test.cpp','run.py']}
def cmd(mode,prefix,ppc,seed,steps=180000,start=70000,every=10,out=10000):
 return [str(R/mode),'--M1','3','--cells','400','--dt','0.005','--ppc',str(ppc),'--seed',str(seed),'--steps',str(steps),'--sample-start',str(start),'--sample-every',str(every),'--out-every',str(out),'--prefix',str(prefix)]
def validate(p,mode,steps,samples):
 z=np.genfromtxt(str(p)+'_final.csv',delimiter=',',names=True)
 assert len(z)==400 and all(np.isfinite(z[n]).all() for n in z.dtype.names)
 assert np.all(z['nsamples']==samples)
 with open(str(p)+'.checkpoint','rb') as f:
  magic=f.readline().strip();sig=f.readline();rng=f.readline();head=f.read(29)
  st,t,stime,sampling,nsamp=struct.unpack('<idd?q',head)
  assert st==steps and nsamp==samples
  assert magic==(b'JFM1439_C4_PAIRFIX_V2' if mode=='corrected' else b'JFM1439_C4_V1')
  lens=[]
  for i in range(21):
   raw=f.read(8);assert len(raw)==8
   n=struct.unpack('<Q',raw)[0];assert n<100000000
   a=np.frombuffer(f.read(n*8),dtype='<f8');assert len(a)==n and np.isfinite(a).all();lens.append(n)
  assert len(set(lens[:4]))==1 and all(n==400 for n in lens[4:]) and not f.read(1)
 return sha(Path(str(p)+'_final.csv'))
def archive(d):
 dest=A/d.relative_to(R);dest.mkdir(parents=True,exist_ok=True);hashes={}
 for p in d.iterdir():
  if not p.is_file() or p.name.endswith('.tmp'):continue
  target=dest/p.name;h=sha(p)
  if not target.exists():shutil.copy2(p,target)
  assert sha(target)==h;hashes[p.name]=h
 (dest/'ARCHIVED.json').write_text(json.dumps(dict(source=str(d),hashes=hashes,time=time.time()),indent=2))
if sys.argv[1]=='gate':
 d=R/'gate';d.mkdir(exist_ok=True)
 for src,exe in [('original.cpp','original'),('corrected.cpp','corrected'),('collision_test.cpp','collision_test')]:subprocess.run(['g++','-O3','-std=c++17',str(R/src),'-o',str(R/exe)],check=True)
 subprocess.run([str(R/'collision_test')],check=True)
 for ppc in [120,960]:
  for mode in ['original','corrected']:
   for label,steps in [('continuous',100),('split',40),('split',100)]:
    p=d/f'{mode}_{ppc}_{label}'
    subprocess.run(cmd(mode,p,ppc,993,steps,20,2,20),check=True)
   assert validate(d/f'{mode}_{ppc}_continuous',mode,100,41)==validate(d/f'{mode}_{ppc}_split',mode,100,41)
  reject=d/f'reject_{ppc}';shutil.copy2(d/f'original_{ppc}_continuous.checkpoint',str(reject)+'.checkpoint')
  q=subprocess.run(cmd('corrected',reject,ppc,993,100,20,2,20),capture_output=True,text=True)
  assert q.returncode!=0 and 'Checkpoint mismatch' in q.stderr
 passed=dict(time=time.time(),manifest=manifest(),restart='bitwise each version',foreign_checkpoint='rejected',pair_bijection='exact',collision_conservation_trials=10000,tolerance=1e-12)
 (R/'PASS.json').write_text(json.dumps(passed,indent=2))
 A.mkdir(parents=True,exist_ok=True)
 for name in ['original.cpp','corrected.cpp','collision_test.cpp','run.py','storage.json','PASS.json']:shutil.copy2(R/name,A/name);assert sha(R/name)==sha(A/name)
 print('GATE PASSED',flush=True)
else:
 passed=json.loads((R/'PASS.json').read_text());assert passed['manifest']==manifest()
 idx=int(os.environ['SLURM_ARRAY_TASK_ID']);ppc=[120,960][idx//2];seed=idx%2+1
 def one(mode):
  d=R/'outputs'/f'{mode}_ppc{ppc}_seed{seed}';d.mkdir(parents=True,exist_ok=True);p=d/'M3'
  if not (d/'DONE.json').exists():
   with (d/'run.log').open('a') as log:subprocess.run(cmd(mode,p,ppc,seed),stdout=log,stderr=subprocess.STDOUT,check=True)
   h=validate(p,mode,180000,11001)
   (d/'DONE.json').write_text(json.dumps(dict(mode=mode,ppc=ppc,seed=seed,steps=180000,cells=400,dt=.005,sample_start=70000,sample_every=10,sha256=h,manifest=manifest(),time=time.time()),indent=2))
  else:validate(p,mode,180000,11001)
  archive(d)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,['original','corrected']))
