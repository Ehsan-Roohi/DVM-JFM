"""Plot completed or explicitly provisional Bird DSMC profiles without smoothing or width fitting."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
fig,axs=plt.subplots(3,4,figsize=(15,10),constrained_layout=True)
keys=['rho','u','theta','sigma','q','m','R','Delta','S','QDelta','Fq','JDelta'];found=0
for d in sorted((R/'runs').glob('M*')):
 if not (d/'profiles.npz').exists():continue
 z=np.load(d/'profiles.npz');x=z['x'];rho=z['rho'];target=.5*(rho[:40].mean()+rho[-40:].mean());cross=np.flatnonzero((rho[:-1]-target)*(rho[1:]-target)<=0)
 xs=float(x[cross[np.argmin(abs(x[cross]))]]) if len(cross) else 0.
 for ax,k in zip(axs.flat,keys):ax.plot(x-xs,z[k],label=d.name);ax.set_title(k);ax.set_xlabel('(x-x_mid)/L0');ax.grid(alpha=.2)
 found+=1
if found:
 axs.flat[0].legend();out=R/'figures';out.mkdir(exist_ok=True)
 fig.suptitle('Bird VHS DSMC: continuum central moments; finite sampling; midpoint translation only')
 for ext in ['png','pdf','svg']:fig.savefig(out/f'dsmc_M2_M5.{ext}',dpi=180)
print(json.dumps({'available_cases':found,'figures_generated':bool(found)}))
