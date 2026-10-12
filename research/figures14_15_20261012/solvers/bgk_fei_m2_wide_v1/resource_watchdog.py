"""Native Linux resource gate; signal ONLY the owned cooperative child."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
GIB=1024**3
def memory():
    info=dict((s.split(':',1)[0],s.split(':',1)[1].strip()) for s in Path('/proc/meminfo').read_text().splitlines())
    host=int(info['MemAvailable'].split()[0])*1024
    if host<=0:raise RuntimeError('Invalid native host memory sample')
    cg=Path('/proc/self/cgroup').read_text().splitlines();mounts=Path('/proc/self/mountinfo').read_text().splitlines()
    modes=[]
    for row in cg:
        _,controllers,loc=row.split(':',2)
        for mi in mounts:
            pre,post=mi.split(' - ',1);a=pre.split();b=post.split()
            if (not controllers and b[0]=='cgroup2') or ('memory' in controllers.split(',') and b[0]=='cgroup' and 'memory' in b[2].split(',')):
                root=Path(a[3]);point=Path(a[4]);relative=Path(loc).relative_to(root);modes.append((point,point/relative,b[0]))
    if not modes:raise RuntimeError('Memory cgroup topology unavailable')
    available=[host];finite=[]
    for mount,current,kind in modes:
        while True:
            lim=current/('memory.max' if kind=='cgroup2' else 'memory.limit_in_bytes')
            use=current/('memory.current' if kind=='cgroup2' else 'memory.usage_in_bytes')
            if lim.exists():
                limit=lim.read_text().strip();usage=int(use.read_text().strip())
                if usage<0:raise RuntimeError('Invalid native cgroup usage')
                if limit!='max':
                    cap=int(limit)
                    if cap<=0:raise RuntimeError('Invalid native cgroup limit')
                    if cap<2**60:available.append(max(0,cap-usage));finite.append({'path':str(current),'limit':cap,'usage':usage})
            if current==mount:break
            if mount not in current.parents:raise RuntimeError('Invalid cgroup ancestry')
            current=current.parent
    return {'available_bytes':min(available),'host_available_bytes':host,'finite_ancestors':finite}
def save(p,data):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(t,p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--script',required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
    receipt=Path(f'RESOURCE_{a.index}.json')
    if receipt.exists():raise RuntimeError('Exclusive launch receipt already exists')
    first=memory()
    if first['available_bytes']<4*GIB:
        save(receipt,{'state':'HELD_RAM_BELOW_4GiB','sample':first});return 2
    child=subprocess.Popen([sys.executable,'-u',a.script,'--index',str(a.index)])
    interrupted=[False]
    def halt(*_):
        interrupted[0]=True
        if child.poll() is None:child.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1,halt);signal.signal(signal.SIGTERM,halt)
    state={'state':'PROCESS_STARTED','pid':child.pid,'start_sample':first,'minimum_available_bytes':first['available_bytes']}
    save(receipt,state)
    while child.poll() is None:
        try:
            sample=memory();state['last_sample']=sample
            state['minimum_available_bytes']=min(state['minimum_available_bytes'],sample['available_bytes'])
            if sample['available_bytes']<GIB:state['stop_reason']='RAM_BELOW_1GiB';halt()
        except Exception as e:state['stop_reason']='INVALID_NATIVE_RAM_SAMPLE';state['sample_error']=str(e);halt()
        save(receipt,state)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:pass
    state.update(state='TERMINAL',exit_code=child.returncode,cooperative_stop=interrupted[0]);save(receipt,state)
    return child.returncode
if __name__=='__main__':sys.exit(main())
