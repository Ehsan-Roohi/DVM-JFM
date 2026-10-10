from pathlib import Path
import hashlib, json, string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'mathtext.fontset':'dejavuserif','font.size':11,'axes.labelsize':12,
 'axes.titlesize':14,'xtick.labelsize':10,'ytick.labelsize':10,
 'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
blue='#0072B2';orange='#D55E00';black='#171717'
fig,axs=plt.subplots(4,3,figsize=(9.2,10.7),sharex='col',
 gridspec_kw={'height_ratios':[1,1,1,1.25]})
fig.subplots_adjust(left=.115,right=.985,top=.955,bottom=.105,hspace=.26,wspace=.29)
meta={'source':'Original continuum-reference exact-heatflux-budget profile CSVs',
 'coordinate':'x-x_s; x is the archived xmfp coordinate, x_s=x[argmax(abs(gradient(rho_file,x)))] as in original plotting script',
 'processing':'No smoothing, interpolation, amplitude normalization, or plateau subtraction. S uses reference column, not budget reconstruction. Delta/3 is a plotting rescaling.',
 'cases':{}}
for j,M in enumerate([2,3,5]):
 path=ROOT/'inputs'/f'M{M}_exact_budget_xmfp_exact_heatflux_budget_profiles.csv'
 d=np.genfromtxt(path,delimiter=',',names=True)
 x=d['x'];q=d['qx'];R=d['Rxx_closure'];D=d['Delta'];S=d['S_R_plus_Delta3_ref']
 assert all(np.isfinite(a).all() for a in [x,q,R,D,S]) and np.all(np.diff(x)>0)
 assert np.allclose(S,R+D/3,rtol=1e-13,atol=1e-13)
 xs=float(x[np.argmax(np.abs(np.gradient(d['rho_file'],x)))])
 xc=x-xs
 series=[q,R,D,S];colors=[black,blue,orange,black]
 for i in range(4):
  ax=axs[i,j]
  ax.axhline(0,color='#aaaaaa',lw=.65,zorder=0)
  if i<3:
   ax.plot(xc,series[i],color=colors[i],lw=1.85)
   lo,hi=series[i].min(),series[i].max()
  else:
   ax.plot(xc,S,color=black,lw=2.1,label=r'$S=R^{\mathrm{cl}}_{xx}+\Delta/3$',zorder=4)
   ax.plot(xc,R,color=blue,lw=1.9,ls=(0,(5,2.5)),label=r'$R^{\mathrm{cl}}_{xx}$',zorder=5)
   ax.plot(xc,D/3,color=orange,lw=1.85,ls=(0,(4,1.8,1,1.8)),label=r'$\Delta/3$',zorder=3)
   lo=min(R.min(),(D/3).min(),S.min());hi=max(R.max(),(D/3).max(),S.max())
  pad=.13*(hi-lo);ax.set_ylim(lo-pad,hi+pad)
  ax.set_xlim(xc[0],xc[-1]);ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
  ax.grid(color='#dedede',linewidth=.55,alpha=.65)
  ax.set_axisbelow(True);ax.tick_params(direction='in',top=True,right=True,length=3)
  ax.text(.045,.07,f'({string.ascii_lowercase[i*3+j]})',transform=ax.transAxes,fontweight='bold',fontsize=10,
          bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
  if i==0:ax.set_title(rf'$M_1={M}$',pad=9)
  if i==3:ax.set_xlabel(r'$(x-x_s)/\lambda_1$')
  if j==0:ax.set_ylabel([r'$q_x$',r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta$',r'$R^{\mathrm{cl}}_{xx},\;\Delta/3,\;S$'][i],labelpad=8)
 handles=[Line2D([],[],color=blue,lw=1.9,ls=(0,(5,2.5))),Line2D([],[],color=orange,lw=1.85,ls=(0,(4,1.8,1,1.8))),Line2D([],[],color=black,lw=2.1)]
 meta['cases'][f'M{M}']={'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(x),'x_s':xs,
 'max_abs_channel_identity_error':float(np.max(np.abs(S-R-D/3))),
 'q_min':float(q.min()),'R_peak':float(R.max()),'Delta_peak':float(D.max()),'S_peak':float(S.max())}
 np.savetxt(OUT/f'M{M}_plotted.csv',np.c_[xc,q,R,D,D/3,S],delimiter=',',header='x_minus_xs_over_lambda1,qx,Rxx_closure,Delta,Delta_over_3,S_reference',comments='')
fig.legend(handles,[r'Tensorial: $R^{\mathrm{cl}}_{xx}$',r'Scalar contribution: $\Delta/3$',r'Combined: $S$'],
 loc='lower center',bbox_to_anchor=(.54,.022),ncol=3,frameon=False,handlelength=3.2,columnspacing=1.8,fontsize=11)
fig.savefig(OUT/'fig1_dvm_reference_overlay.pdf',bbox_inches='tight',pad_inches=.07)
fig.savefig(OUT/'fig1_dvm_reference_overlay.png',dpi=350,bbox_inches='tight',pad_inches=.07)
fig.savefig(OUT/'fig1_dvm_reference_overlay.svg',bbox_inches='tight',pad_inches=.07)
fig.savefig(OUT/'preview.png',dpi=130,bbox_inches='tight',pad_inches=.07)
plt.close(fig)
(OUT/'provenance.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
