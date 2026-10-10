from pathlib import Path
import json,hashlib,string,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
ROOT=Path.cwd();OUT=ROOT/'output/pdf/shock_comparisons_20261007';OUT.mkdir(parents=True,exist_ok=True)
B=ROOT/'revision_runs/dvm_r26_transport_match_20261007/completed_analysis'
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':11,'axes.labelsize':12,'axes.titlesize':13,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.8})
BLUE='#0072B2';RED='#D55E00';GOLD='#E69F00';GREEN='#009E73';PURPLE='#8C56A2';BLACK='#20252B'
fields=['rho','u','T','q','sigma','R','Delta','S'];labels=[r'$\rho$',r'$u_x$',r'$T$',r'$q_x$',r'$\sigma_{xx}$',r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta$',r'$S=R^{\mathrm{cl}}_{xx}+\Delta/3$']
meta={'sources':{},'figures':{},'processing':'No smoothing or plateau subtraction; matched comparisons use density-midpoint translation only, no width fitting.'}
def read_npz(path):
 meta['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest();a=np.load(path,allow_pickle=False);d={k:a[k].copy() for k in a.files};assert all(np.isfinite(v).all() for v in d.values() if v.dtype.kind in 'fci');return d
D={}
for name in ['M2_base','M2_x640','M2_x1280','M2_v641','M2_vextent','M2_L45','M2_dt_half']:
 a=read_npz(B/'DVM_profiles'/name/'profiles.npz');assert a['columns'].tolist()==fields;D[name]=(a['x'],a['fields'])
a=read_npz(B/'R26_M2_N641_profile.npz');R=(a['x'],np.column_stack([a['theta'] if k=='T' else a[k] for k in fields]))
def align(x,f):
 mid=(1+16/7)/2;ii=np.flatnonzero((f[:-1,0]-mid)*(f[1:,0]-mid)<=0);assert len(ii)==1;i=ii[0];xs=x[i]+(mid-f[i,0])*(x[i+1]-x[i])/(f[i+1,0]-f[i,0]);return x-xs,f
D={k:align(*v) for k,v in D.items()};R=align(*R)
def style(ax,letter=None):
 ax.grid(color='#DDE3E8',lw=.65,alpha=.8);ax.set_axisbelow(True);ax.axhline(0,color='#9BA5AE',lw=.7,zorder=0);ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.tick_params(direction='out',length=3)
 if letter:ax.text(.03,.95,letter,transform=ax.transAxes,va='top',fontweight='bold',fontsize=11,bbox=dict(fc='white',ec='none',alpha=.85,pad=1.5))
def header(fig,title,subtitle):
 fig.suptitle(title,fontsize=19,fontweight='bold',y=.987);fig.text(.5,.947,subtitle,ha='center',fontsize=11,color='#4D5964')
def save(fig,name,title):
 fig.savefig(OUT/(name+'.png'),dpi=210,facecolor='white');fig.savefig(OUT/(name+'.pdf'),facecolor='white');fig.savefig(OUT/(name+'.svg'),facecolor='white');collection.savefig(fig);meta['figures'][name]={'title':title};plt.close(fig)
collection=PdfPages(OUT/'first_figure_working.pdf')
# 1: exact matched-transport comparison on native grids
fig,axs=plt.subplots(2,4,figsize=(15,8));fig.subplots_adjust(left=.065,right=.985,bottom=.12,top=.825,hspace=.36,wspace=.35)
header(fig,'Mach 2: R26 versus transport-matched DVM','Same Pr = 2/3, viscosity law, far-field states and physical length unit')
x=np.linspace(-15,15,1201);yr=np.column_stack([np.interp(x,R[0],R[1][:,i]) for i in range(8)]);yd=np.column_stack([np.interp(x,D['M2_x1280'][0],D['M2_x1280'][1][:,i]) for i in range(8)])
expected=json.loads((B/'analysis.json').read_text())['R26_vs_Shakhov_DVM']['metrics']
for i,ax in enumerate(axs.flat):
 ax.plot(*[R[0],R[1][:,i]],color=BLUE,lw=2.25,label='R26, 641 nodes');ax.plot(D['M2_x1280'][0],D['M2_x1280'][1][:,i],color=RED,lw=2,ls=(0,(5,2)),label='Shakhov DVM, 1280 cells')
 error=100*np.linalg.norm(yr[:,i]-yd[:,i])/np.linalg.norm(yd[:,i]);assert abs(error-expected[fields[i]]['relative_L2_percent_using_DVM_norm'])<1e-10
 ax.set_title(labels[i]+'\n'+rf'$L_2$ difference: {error:.2f}%',pad=8,fontsize=12);style(ax,f'({string.ascii_lowercase[i]})');ax.set_xlim(-15,15)
 if i>=4:ax.set_xlabel(r'$(x-x_s)/\ell_1$')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.935),ncol=2,frameon=False,fontsize=12,handlelength=3)
fig.text(.5,.047,'Different collision operators; differences are not pure R26 errors. DVM profiles at t = 400; stationarity criterion not met.',ha='center',fontsize=10,color='#52616E')
fig.text(.5,.018,r'$\ell_1=\mu_1/(\rho_1\sqrt{T_1})$; density-midpoint alignment only. Percentages use the DVM profile norm on $[-15,15]$.',ha='center',fontsize=10,color='#52616E')
save(fig,'01_M2_R26_vs_DVM','Mach 2 R26 vs transport-matched DVM')

collection.close()
