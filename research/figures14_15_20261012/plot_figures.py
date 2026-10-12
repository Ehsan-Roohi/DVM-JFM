from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;V=P.parent;I=P/'inputs';C=P/'data';C.mkdir(exist_ok=True)
keys=['rho','T','q','R','Delta','S'];labels=[r'$\rho$',r'$T$',r'$q_x$',r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta$',r'$S$']
models=['DSMC','BGK','Shakhov DVM','Shakhov R26','ES-BGK','ES-FP']
palette={'DSMC':('#8e8e8e','-',1.0),'BGK':('#111111','-',2.8),'Shakhov DVM':('#0072B2','--',2.5),'Shakhov R26':('#D55E00','-.',2.5),'ES-BGK':('#882255',':',3.0),'ES-FP':('#009E73',(0,(6,2,1,2)),2.8)}
names={'DSMC':'DSMC','BGK':r'BGK: $Pr=1$','Shakhov DVM':r'Shakhov DVM: $Pr=2/3$','Shakhov R26':r'Shakhov R26: $Pr=2/3$','ES-BGK':r'ES-BGK: $Pr=2/3$','ES-FP':r'ES-FP: $Pr=2/3$'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data={m:{} for m in [2,3,4,5]}
for m in data:
 for model in models:
  path=C/f'M{m}_{model.replace(" ","_")}.csv'
  if not path.exists(): continue
  z=np.genfromtxt(path,delimiter=',',names=True)
  assert np.isfinite(np.column_stack([z[k] for k in keys])).all()
  assert np.allclose(z['S'],z['R']+z['Delta']/3,rtol=1e-10,atol=1e-9)
  data[m][model]=(z['x_density_midpoint'],{k:z[k] for k in keys})
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':20,'axes.labelsize':24,'axes.titlesize':25,'legend.fontsize':17,'xtick.labelsize':18,'ytick.labelsize':18,'pdf.fonttype':42})
fig,ax=plt.subplots(3,2,figsize=(12.6,12.7));fig.subplots_adjust(left=.105,right=.98,bottom=.10,top=.79,wspace=.30,hspace=.39)
for j,(a,k,l) in enumerate(zip(ax.flat,keys,labels)):
 for model in models:
  x,d=data[2][model];color,ls,lw=palette[model];label=names[model]
  a.plot(x,d[k],label=label,color=color,ls=ls,lw=lw)
 a.set_xlim(-15,15);a.grid(alpha=.20);a.set_axisbelow(True);a.set_ylabel(l);a.set_xlabel(r'$(x-x_s)/\lambda_1$');a.text(.015,1.035,'('+chr(97+j)+')',transform=a.transAxes,va='bottom',fontsize=21)
fig.suptitle(r'Mach 2: transport-matched shock profiles ($\mu=T$)',y=.99,fontsize=25)
fig.legend(*ax[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,frameon=False,columnspacing=1.45,handlelength=3.1,labelspacing=.7)
fig.savefig(P/'Figure14_clean_labels.pdf',bbox_inches='tight');fig.savefig(P/'Figure14_clean_labels.png',dpi=160,bbox_inches='tight');plt.close(fig)
plt.rcParams.update({'font.size':22,'axes.labelsize':28,'axes.titlesize':28,'legend.fontsize':22,'xtick.labelsize':21,'ytick.labelsize':21})
fig,ax=plt.subplots(6,3,figsize=(17.5,20));fig.subplots_adjust(left=.09,right=.985,top=.84,bottom=.125,wspace=.29,hspace=.32)
for j,m in enumerate([3,4,5]):
 for i,(key,label) in enumerate(zip(keys,labels)):
  a=ax[i,j]
  for model in models:
   if model not in data[m]:continue
   x,d=data[m][model];color,ls,lw=palette[model];a.plot(x,d[key],label=names[model],color=color,ls=ls,lw=lw)
  a.set_xlim(-15,15);a.grid(alpha=.2);a.set_axisbelow(True)
  if j==0:a.set_ylabel(label)
  if i==0:a.set_title(rf'$M_1={m}$',pad=18)
  if i==5:a.set_xlabel(r'$(x-x_s)/\lambda_1$')
  else:a.tick_params(labelbottom=False)
fig.suptitle(r'Transport-matched shock profiles ($\mu=T$)',y=.992,fontsize=32)
fig.legend(*ax[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.962),ncol=3,frameon=False,handlelength=3.2,columnspacing=1.4,labelspacing=.8)
fig.savefig(P/'Figure15_clean_labels.pdf',bbox_inches='tight');fig.savefig(P/'Figure15_clean_labels.png',dpi=150,bbox_inches='tight');plt.close(fig)
