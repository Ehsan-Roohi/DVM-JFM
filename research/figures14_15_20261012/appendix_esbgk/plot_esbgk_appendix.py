from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
cols=['rho','T','q','R','Delta','S']
labels=[r'$\rho$',r'$T$',r'$q_x$',r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta$',r'$S$']
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':20,'axes.labelsize':24,'legend.fontsize':17,'xtick.labelsize':18,'ytick.labelsize':18,'pdf.fonttype':42})
for m in (2,3):
    paths=[(P/f'runs/M{m}_base/profiles.npz','ES-BGK: baseline, t=400','#56B4E9',':'),(P/f'runs/M{m}_v2/profiles.npz','ES-BGK: refined, t=400','#009E73','-'),(P/f'comparison_inputs/M{m}/DSMC_2000ppc_M{m}.npz','DSMC: 2000 PPC','#888888','-'),(P/f'comparison_inputs/M{m}/DVM_Shakhov_M{m}.npz','Shakhov DVM','#0072B2','--'),(P/f'comparison_inputs/M{m}/R26_accepted_Shakhov_M{m}.npz','Shakhov R26','#D55E00','-.')]
    fig,axs=plt.subplots(3,2,figsize=(12.6,12.7))
    fig.subplots_adjust(left=.105,right=.98,top=.86,bottom=.085,hspace=.36,wspace=.30)
    for p,name,color,ls in paths:
        d=dict(np.load(p,allow_pickle=False))
        if 'fields' in d:d.update({str(k):d['fields'][:,i] for i,k in enumerate(d['columns'])})
        if 'theta' in d:d['T']=d['theta']
        rho2=4*m*m/(m*m+3);y=d['rho']-(1+rho2)/2;x=d['x']
        ii=np.flatnonzero(y[:-1]*y[1:]<=0);assert len(ii)>0
        i=ii[np.argmin(abs(x[ii]))];xs=x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])
        for ax,k,lab in zip(axs.flat,cols,labels):
            ax.plot(x-xs,d[k],color=color,ls=ls,lw=1.1 if 'DSMC' in name else 2.7,label=name)
            ax.set(xlim=(-15,15),ylabel=lab,xlabel=r'$(x-x_s)/\lambda_1$');ax.grid(alpha=.18)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.987),ncol=2,frameon=False,labelspacing=.65,handlelength=3)
    fig.savefig(P/f'ESBGK_M{m}_comparison.pdf',bbox_inches='tight')
    fig.savefig(P/f'ESBGK_M{m}_comparison.png',dpi=130,bbox_inches='tight')
    plt.close(fig)
