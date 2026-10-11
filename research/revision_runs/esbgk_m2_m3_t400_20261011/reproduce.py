"""Recompute published ES-BGK budgets and comparison figures from frozen inputs."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import esbgk as es
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reproduced';OUT.mkdir(exist_ok=True)
checks={}
for d in sorted((ROOT/'runs').iterdir()):
 st=json.loads((d/'STATUS.json').read_text());c=st['config']
 for name,sha in st['sha256'].items():assert es.ref.sha(d/name)==sha
 z=np.load(d/'checkpoint.npz',allow_pickle=False)
 sig=hashlib.sha256((json.dumps(c,sort_keys=True)+es.ref.sha(ROOT/'esbgk.py')+es.ref.sha(ROOT/'shakhov_reference.py')).encode()).hexdigest()
 assert sig==st['signature']==str(z['signature'])
 x,v,w,_,_,_=es.ref.setup(c);out=OUT/d.name;out.mkdir(exist_ok=True)
 report=es.budgets(z['f'],x,v,w,c,out)
 for key in ['S_relative_L2','Delta_relative_L2','R_relative_L2']:
  assert abs(report[key]-st['budgets'][key])<1e-12
 checks[d.name]={key:report[key] for key in ['S_relative_L2','Delta_relative_L2','R_relative_L2']}
(OUT/'RECOMPUTATION_CHECK.json').write_text(json.dumps({'PASS':True,'errors':checks},indent=2))
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':15,'axes.labelsize':18,'legend.fontsize':13,'xtick.labelsize':14,'ytick.labelsize':14,'pdf.fonttype':42})
cols=['rho','T','q','R','Delta','S'];labels=[r'$\rho$',r'$T$',r'$q_x$',r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta$',r'$S$'];provenance=[]
for m in [2,3]:
 paths=[(ROOT/f'runs/M{m}_base/profiles.npz','ES-BGK: baseline, t=400','#56B4E9',':'),(ROOT/f'runs/M{m}_v2/profiles.npz','ES-BGK: refined, t=400','#009E73','-'),(ROOT/f'comparison_inputs/M{m}/DSMC_2000ppc_M{m}.npz','DSMC: 2000 PPC, raw','#888888','-'),(ROOT/f'comparison_inputs/M{m}/DVM_Shakhov_M{m}.npz','Shakhov DVM: snapshot','#0072B2','--'),(ROOT/f'comparison_inputs/M{m}/R26_accepted_Shakhov_M{m}.npz','Corrected Shakhov R26','#D55E00','-.')]
 fig,axs=plt.subplots(3,2,figsize=(10,10));fig.subplots_adjust(left=.11,right=.98,top=.84,bottom=.15,hspace=.43,wspace=.30)
 for p,label,color,ls in paths:
  d=dict(np.load(p,allow_pickle=False))
  if 'fields' in d:d.update({str(k):d['fields'][:,i] for i,k in enumerate(d['columns'])})
  if 'theta' in d:d['T']=d['theta']
  rho2=4*m*m/(m*m+3);y=d['rho']-(1+rho2)/2;x=d['x'];ii=np.flatnonzero(y[:-1]*y[1:]<=0);assert len(ii)>0
  i=ii[np.argmin(abs(x[ii]))];xs=x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i]);xx=x-xs
  for ax,k,lab in zip(axs.flat,cols,labels):ax.plot(xx,d[k],color=color,ls=ls,lw=1.1 if 'DSMC' in label else 2,label=label);ax.set(xlim=(-15,15),ylabel=lab,xlabel=r'$(x-x_s)/\lambda_1$');ax.grid(alpha=.18)
  provenance.append({'mach':m,'path':p.relative_to(ROOT).as_posix(),'sha256':es.ref.sha(p),'density_midpoint_shift':float(xs),'crossings':len(ii)})
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.935),ncol=2,frameon=False)
 fig.suptitle(f'Mach {m}: finite-time ES-BGK and archived reference profiles',y=.99,fontsize=18)
 fig.text(.5,.025,'ES-BGK: Pr=2/3, t=400; stationarity criterion NOT passed.\nDensity-midpoint alignment only; different models and grids, not a grid-error study.',ha='center',fontsize=12)
 fig.savefig(OUT/f'ESBGK_M{m}_comparison.pdf');fig.savefig(OUT/f'ESBGK_M{m}_comparison.png',dpi=150);plt.close(fig)
(OUT/'FIGURE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
print('PASS: all six budgets reproduced; both Mach comparison figures generated')
