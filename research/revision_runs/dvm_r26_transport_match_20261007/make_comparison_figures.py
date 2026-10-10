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
collection=PdfPages(OUT/'all_shock_comparisons.pdf')
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
# 2: three-grid profile overlays
fig,axs=plt.subplots(2,2,figsize=(11.5,8.5));fig.subplots_adjust(left=.09,right=.975,bottom=.12,top=.81,hspace=.32,wspace=.25)
header(fig,'Mach 2 DVM: three spatial grids','320, 640 and 1280 cells; native profiles at the same final physical time')
grid=json.loads((B/'three_grid_report.json').read_text())['groups']['DVM_Shakhov']['metrics']
for j,(name,color,ls) in enumerate([('M2_base',GOLD,(0,(1,1.5))),('M2_x640',RED,(0,(5,2))),('M2_x1280',BLUE,'-')]):
 for ax,i in zip(axs.flat,[3,5,6,7]):ax.plot(D[name][0],D[name][1][:,i],color=color,ls=ls,lw=2.3 if j==2 else 2,label=['320 cells','640 cells','1280 cells'][j])
for j,(ax,i) in enumerate(zip(axs.flat,[3,5,6,7])):
 style(ax,f'({string.ascii_lowercase[j]})');ax.set_title(labels[i]);ax.set_xlim(-15,15);ax.set_xlabel(r'$(x-x_s)/\ell_1$');ax.text(.97,.94,f"640 vs 1280: {grid[fields[i]]['medium_fine_percent']:.2f}%",ha='right',va='top',transform=ax.transAxes,fontsize=11,bbox=dict(fc='white',ec='none',alpha=.87))
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.89),ncol=3,frameon=False,fontsize=12,handlelength=3)
fig.text(.5,.045,'Percentages are aligned relative L2 differences. Spatial refinement also reduces the CFL time step.',ha='center',fontsize=10,color='#52616E');fig.text(.5,.019,'Final-time sensitivity: the original stationarity criterion is not met; no grid-independence claim is implied.',ha='center',fontsize=10,color='#52616E')
save(fig,'02_M2_DVM_three_grids','Mach 2 DVM three spatial grids')
# 3: all four non-spatial sensitivity cases (all seven DVM runs covered)
fig,axs=plt.subplots(2,4,figsize=(15,8));fig.subplots_adjust(left=.075,right=.985,bottom=.13,top=.80,hspace=.31,wspace=.34)
header(fig,'Mach 2 DVM: velocity, domain and time-step sensitivity','Top: profile overlays. Bottom: deviations from the baseline, normalized by its peak absolute value.')
cases=[('M2_base','Baseline',BLACK,'-'),('M2_v641','Velocity grid doubled',BLUE,(0,(5,2))),('M2_vextent','Velocity extent increased',RED,(0,(3,2))),('M2_L45','Spatial domain increased',GREEN,(0,(1,2))),('M2_dt_half','Time step halved',PURPLE,(0,(6,2,1,2)))]
for name,label,color,ls in cases:
 for j,i in enumerate([3,5,6,7]):
  xs,f=D[name];axs[0,j].plot(xs,f[:,i],label=label,color=color,ls=ls,lw=1.9)
  if name!='M2_base':
   xb,fb=D['M2_base'];mask=(xs>=-15)&(xs<=15);delta=100*(f[mask,i]-np.interp(xs[mask],xb,fb[:,i]))/np.max(abs(fb[:,i]));axs[1,j].plot(xs[mask],delta,color=color,ls=ls,lw=1.7)
for j,i in enumerate([3,5,6,7]):
 for row in [0,1]:style(axs[row,j]);axs[row,j].set_xlim(-15,15)
 axs[0,j].set_title(labels[i]);axs[1,j].set_xlabel(r'$(x-x_s)/\ell_1$');axs[1,j].set_ylabel('Peak-normalized change (%)',fontsize=10)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,frameon=False,fontsize=11,handlelength=3)
fig.text(.5,.046,'Separate parameter variations around the 320-cell baseline; the spatial-grid comparison is shown in Figure 2.',ha='center',fontsize=10,color='#52616E');fig.text(.5,.019,'Velocity-extent variation also changes the CFL time step. All profiles are at t = 400; stationarity criterion not met.',ha='center',fontsize=10,color='#52616E')
save(fig,'03_M2_DVM_parameter_sensitivity','Mach 2 DVM velocity/domain/time-step sensitivity')
# 4: lower-Mach historical BGK cases, original reference fields, no R26 overlay
fig,axs=plt.subplots(4,3,figsize=(11,12));fig.subplots_adjust(left=.09,right=.985,bottom=.105,top=.875,hspace=.27,wspace=.30)
header(fig,'Lower-Mach BGK DVM: M = 2, 3 and 5','Constant relaxation time, Pr = 1; archived continuum-reference fourth moments')
for col,mach in enumerate([2,3,5]):
 p=ROOT/f'figure1_redesign_20261007/output/M{mach}_plotted.csv';meta['sources'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();a=np.genfromtxt(p,delimiter=',',names=True);xx=a['x_minus_xs_over_lambda1'];series=[a['qx'],a['Rxx_closure'],a['Delta'],a['S_reference']];assert all(np.isfinite(v).all() for v in series)
 for row,y in enumerate(series):
  ax=axs[row,col];ax.plot(xx,y,color=[BLACK,BLUE,RED,GREEN][row],lw=2);style(ax,f'({string.ascii_lowercase[3*row+col]})');ax.set_xlim(xx[0],xx[-1])
  if col==0:ax.set_ylabel([labels[3],labels[5],labels[6],labels[7]][row])
  if row==0:ax.set_title(rf'$M_1={mach}$',fontsize=15)
  if row==3:ax.set_xlabel(r'$(x-x_s)/\lambda_1$')
fig.text(.5,.058,'Original BGK manuscript cases. These are different from the new Pr = 2/3 transport-matched Shakhov runs.',ha='center',fontsize=10,color='#52616E');fig.text(.5,.028,'No plateau subtraction or amplitude rescaling; each panel retains its physical moment scale.',ha='center',fontsize=10,color='#52616E')
save(fig,'04_lower_Mach_BGK_DVM','Lower-Mach BGK DVM M2/M3/M5')
# 5: higher-Mach validated extraction, full vs lite, raw/discrete distinction
H={}
for tag in ['M6','M8','M12','M8lite','M12lite']:
 p=ROOT/'revision_runs/high_mach_inventory/local_cpu/validated_results'/tag/'moments.npz';H[tag]=read_npz(p)
fig,axs=plt.subplots(4,3,figsize=(11,12));fig.subplots_adjust(left=.10,right=.985,bottom=.12,top=.84,hspace=.28,wspace=.29)
header(fig,'Higher-Mach BGK DVM: M = 6, 8 and 12','Pr = 1; independently extracted full-distribution moments, without plateau correction')
keys=['qx','Rxx_closure','Delta_dvm','S_dvm']
for col,mach in enumerate([6,8,12]):
 for tag,ls,color in [(f'M{mach}','-',BLUE)]+([] if mach==6 else [(f'M{mach}lite',(0,(5,2)),RED)]):
  a=H[tag];xx=a['x_mfp'];xs=float(xx[np.argmax(abs(np.gradient(a['rho'],xx)))]);xx=xx-xs
  for row,key in enumerate(keys):axs[row,col].plot(xx,a[key],lw=2.15 if 'lite' not in tag else 1.8,color=color,ls=ls,label='Full' if 'lite' not in tag else 'Lite')
 for row in range(4):
  ax=axs[row,col];style(ax,f'({string.ascii_lowercase[3*row+col]})');ax.set_xlim(-{6:65,8:80,12:120}[mach],{6:65,8:80,12:120}[mach])
  if col==0:ax.set_ylabel([labels[3],labels[5],r'$\Delta_{\mathrm{discrete}}$',r'$S_{\mathrm{discrete}}$'][row])
  if row==0:ax.set_title(rf'$M_1={mach}$',fontsize=15)
  if row==3:ax.set_xlabel(r'$(x-x_s)/\lambda_1$')
fig.legend(handles=[Line2D([],[],color=BLUE,lw=2.2,label='Full distribution'),Line2D([],[],color=RED,lw=1.9,ls=(0,(5,2)),label='Lite configuration (M8 and M12)')],loc='upper center',bbox_to_anchor=(.5,.902),ncol=2,frameon=False,fontsize=12,handlelength=3)
fig.text(.5,.062,'Shock-region views; fourth moments use the discrete matching-Maxwellian reference. Far-field offsets are retained.',ha='center',fontsize=10,color='#52616E');fig.text(.5,.034,'Full/lite configurations differ in multiple numerical parameters; their agreement is a sensitivity comparison.',ha='center',fontsize=10,color='#52616E');fig.text(.5,.011,'No accepted higher-Mach R26 solution is available for an overlay; these BGK curves do not validate the Shakhov reference model.',ha='center',fontsize=9.5,color='#52616E')
save(fig,'05_higher_Mach_BGK_DVM','Higher-Mach BGK DVM M6/M8/M12')
collection.close();(OUT/'figure_provenance.json').write_text(json.dumps(meta,indent=2))
archive=OUT/'shock_comparison_figures.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():
  if p.suffix in ['.png','.pdf','.svg','.json']:z.write(p,p.name)
print(json.dumps({'output':str(OUT),'figures':list(meta['figures']),'zip':str(archive)},indent=2))
