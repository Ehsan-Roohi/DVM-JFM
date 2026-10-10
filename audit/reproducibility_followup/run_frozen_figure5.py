"""Monitor ONE archived postprocessor on its allocated compute node."""
from pathlib import Path
import json, os, subprocess, sys, time

def ram():
    lines=Path('/proc/meminfo').read_text().splitlines()
    avail=next(int(s.split()[1])*1024 for s in lines if s.startswith('MemAvailable:'))
    remaining=[avail];recognized=False
    for row in Path('/proc/self/cgroup').read_text().splitlines():
        _,controllers,relative=row.split(':',2)
        if controllers=='':
            root=Path('/sys/fs/cgroup');p=root/relative.lstrip('/')
            names=('memory.max','memory.current')
        elif 'memory' in controllers.split(','):
            root=Path('/sys/fs/cgroup/memory');p=root/relative.lstrip('/')
            names=('memory.limit_in_bytes','memory.usage_in_bytes')
        else:continue
        if '..' in Path(relative).parts or not p.is_dir():raise RuntimeError('Invalid cgroup path')
        recognized=True;seen=False
        while p.is_relative_to(root):
            f=p/names[0];g=p/names[1]
            if f.exists():
                if not g.is_file():raise RuntimeError('Missing memory usage')
                seen=True
                limit=f.read_text().strip()
                if limit!='max' and int(limit)<2**60:
                    remaining.append(int(limit)-int(g.read_text().strip()))
            if p==root:break
            p=p.parent
        if not seen:raise RuntimeError('No memory controller files')
    if not recognized:raise RuntimeError('Unsupported cgroup topology')
    return min(remaining)/2**30

root=Path.cwd();start=time.monotonic();first=ram()
if first<4:raise RuntimeError(f'Initial available RAM {first} <4GiB')
env=os.environ.copy()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
cmd=[sys.executable,'sources/postprocess_shakhov_channel_consistency.py','--fullstate','input.npz',
     '--case','M3_sourcefixed_frozen_cpu_regenerated','--outdir','output','--Pr',str(2/3),
     '--Kn','1','--device','cpu','--target-chunk','32','--moment-chunk','32']
with open('stdout.log','w') as out,open('stderr.log','w') as err:
    child=subprocess.Popen(cmd,env=env,stdout=out,stderr=err)
    low=first;failure=None
    while child.poll() is None:
        try:
            current=ram();low=min(low,current)
            if current<1:failure='RAM_BELOW_1GiB'
            if time.monotonic()-start>600:failure='EXTERNAL_600s_LIMIT'
        except Exception as e:failure='INVALID_MEMORY_SAMPLE:'+str(e)
        if failure:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            break
        time.sleep(2)
result={'kind':'FROZEN_SOURCE_POSTPROCESS_ONLY','exit_code':child.returncode,'failure':failure,
        'elapsed_s':time.monotonic()-start,'initial_available_GiB':first,'minimum_available_GiB':low,
        'cpu_count':1,'command':cmd,'no_shock_evolution':True}
Path('EXECUTION_RECEIPT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
if failure or child.returncode:sys.exit(1)
