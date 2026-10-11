"""After all packets are terminal, export comparisons and readable figures."""
import csv, hashlib, json, subprocess, sys, tarfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(ROOT/'compare.py')],check=True)
files=[]
for out in sorted((ROOT/'runs').glob('M*')):
    if not (out/'STATUS.json').exists():continue
    z=np.load(out/'profiles.npz');b=np.load(out/'budgets.npz')
    data=np.column_stack([z['x'],z['fields'],b['S_reconstructed'],b['Delta_reconstructed'],b['R_reconstructed']])
    np.savetxt(out/'fields_and_reconstruction.csv',data,delimiter=',',header='x,rho,u,T,q,sigma,R,Delta,S,S_reconstructed,Delta_reconstructed,R_reconstructed',comments='')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,3,figsize=(12,7))
        for axis,index,label in zip(ax.flat[:3],[0,1,2],['Density','Velocity','Temperature']):
            axis.plot(z['x'],z['fields'][:,index]);axis.set_ylabel(label)
        for axis,index,label,key in zip(ax.flat[3:],[7,6,5],['S','Delta','R'],['S_reconstructed','Delta_reconstructed','R_reconstructed']):
            axis.plot(z['x'],z['fields'][:,index],label='Direct velocity moments');axis.plot(z['x'],b[key],'--',label='Steady-form budget');axis.set_ylabel(label)
        for axis in ax.flat:axis.set_xlabel('x / upstream Shakhov relaxation length');axis.grid(alpha=.2)
        ax.flat[3].legend(fontsize=8)
        status=json.loads((out/'STATUS.json').read_text())
        fig.suptitle(f"ES-BGK {out.name}, Pr=2/3, t={status['time']:.3f}, stationary={status['stationary']}")
        fig.tight_layout();fig.savefig(out/'fields_and_reconstruction.png',dpi=180);fig.savefig(out/'fields_and_reconstruction.pdf');plt.close(fig)
    except ImportError:
        (out/'PLOT_UNAVAILABLE.txt').write_text('Matplotlib missing; CSV export remains complete.')
    files.extend(p for p in out.iterdir() if p.is_file())
files.extend(ROOT/name for name in ['COMPARISON.json','PLAN.md','manifest.json','PASS.json','esbgk.py','shakhov_reference.py','compare.py','finalize.py','run.sbatch','source.sha256'])
receipt={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(ROOT/'CLOSED_MANIFEST.json').write_text(json.dumps(receipt,indent=2))
files.append(ROOT/'CLOSED_MANIFEST.json')
with tarfile.open(ROOT/'closed_results.tar.gz','w:gz') as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)))
(ROOT/'closed_results.sha256').write_text(hashlib.sha256((ROOT/'closed_results.tar.gz').read_bytes()).hexdigest()+'  closed_results.tar.gz\n')
