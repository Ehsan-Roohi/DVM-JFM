from pathlib import Path
import json,hashlib,sys,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdfplumber
V=Path(__file__).resolve().parent.parent; A=V.parent/'shock_manuscript_audit_v19_20261010'; P=A/'publication'; R=P/'research'; RUN=R/'revision_runs'; F=A/'followup'
OUT=V/'figures'; IN=V/'figure_reproduction/inputs';IN.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True); (V/'figure_reproduction/preview').mkdir(parents=True,exist_ok=True)
meta={}; C=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#222222']
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':14,'axes.labelsize':15,'axes.titlesize':16,'legend.fontsize':12,'xtick.labelsize':12,'ytick.labelsize':12,'lines.linewidth':1.8,'pdf.fonttype':42})
def inp(p,name=None):
 p=Path(p); name=name or p.name; raw=(IN/name).read_bytes()
 q=IN/name;q.write_bytes(raw);meta[name]={'original_path':str(p),'sha256':hashlib.sha256(raw).hexdigest()};return q
def z(p,name=None):return np.load(inp(p,name),allow_pickle=False)
def csv(p,name=None):return np.genfromtxt(inp(p,name),delimiter=',',names=True)
def save(fig,n):
 fig.savefig(OUT/(n+'.pdf'),bbox_inches='tight');fig.savefig(V/'figure_reproduction/preview'/('new_'+n+'.png'),dpi=120,bbox_inches='tight');plt.close(fig)
def style(ax,title=None,xlabel=None):
 ax.grid(alpha=.18);ax.set_axisbelow(True)
 ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
 ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
 if title:ax.set_title(title,pad=8)
 if xlabel:ax.set_xlabel(xlabel)
def legend(fig,ax,ncol=2):fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.01),ncol=ncol,frameon=False)
XL=r'$(x-x_s)/\lambda_{\mathrm{rel},1}$'; XH=r'$x/\lambda_{\mathrm{rel},1}$'; XD=r'$(x-x_s)/\lambda_1$'
labs=[r'$\rho$',r'$T$',r'$q_x$',r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta_d$',r'$S_d$']
# Figure1: exact archived continuum-reference moments, maximum-gradient centring.
fig,ax=plt.subplots(4,3,figsize=(8,9));fig.subplots_adjust(left=.10,right=.98,bottom=.13,top=.93,wspace=.40,hspace=.30)
for j,m in enumerate([2,3,5]):
 d=csv(R/f'figure1_redesign_20261007/inputs/M{m}_exact_budget_xmfp_exact_heatflux_budget_profiles.csv',f'fig1_M{m}.csv');x=d['x'];xs=x[np.argmax(abs(np.gradient(d['rho_file'],x)))];x=x-xs
 ys=[d['qx'],d['Rxx_closure'],d['Delta']]
 for i,y in enumerate(ys):ax[i,j].plot(x,y,color=['black',C[0],C[1]][i]);style(ax[i,j]);ax[i,j].set_ylabel([r'$q_x$',r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta_c$'][i] if j==0 else '')
 for y,col,ls,label in [(d['Rxx_closure'],C[0],'--',r'$R^{\mathrm{cl}}_{xx}$'),(d['Delta']/3,C[1],'-.',r'$\Delta_c/3$'),(d['S_R_plus_Delta3_ref'],'black','-',r'$S$')]:ax[3,j].plot(x,y,color=col,ls=ls,label=label)
 ax[0,j].set_title(rf'$M_1={m}$');ax[3,j].set_xlabel(XL);style(ax[3,j])
fig.legend(*ax[3,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);save(fig,'fig1_dvm_reference')
# Figure5: only postprocessed stored reductions; no distribution/target re-evaluation.
fig,ax=plt.subplots(3,3,figsize=(8,7.4));fig.subplots_adjust(left=.09,right=.98,top=.85,bottom=.10,wspace=.42,hspace=.44)
def plateau(y):n=max(5,int(.08*len(y)));return y-np.mean(np.r_[y[:n],y[-n:]])
for i,m in enumerate([2,3,5]):
 d=z(F/f'BGK_M{m}_regenerated/reductions.npz',f'budget_M{m}.npz');b=z(F/f'BGK_M{m}_regenerated/scalar_window7_reconciliation.npz',f'scalar_M{m}.npz');p='recomputed_macros_';x=d['x'];D=plateau(d[p+'D']);rr=plateau(d[p+'R']);S=plateau(d[p+'S_rec']);db=plateau(b[p+'D_rec_window7'])
 for j,(ref,rec) in enumerate([(D,db),(rr,S),(rr,S-db/3)]):
  ax[i,j].plot(x,ref,'k-',label='Direct moments');ax[i,j].plot(x,rec,color=C[1],ls='--',label='Budget reconstruction');style(ax[i,j]);lim={2:20,3:30,5:40}[m];ax[i,j].set_xlim(-lim,lim)
  if j==2:ax[i,j].plot(x,S-D/3,color=C[0],ls=':',label=r'Complete-$\Delta$ input')
  if j==0:ax[i,j].set_ylabel(rf'$M_1={m}$'+'\n'+r'$\Delta_c$')
  else:ax[i,j].set_ylabel(r'$R^{\mathrm{cl}}_{xx}$')
  if i==2:ax[i,j].set_xlabel(XH)
for j,t in enumerate(['Scalar budget','Heat channel only','Two budgets']):ax[0,j].set_title(t)
legend(fig,ax[0,2],2);save(fig,'fig3_two_budget_recovery')
# Figure6: original raw stored coordinates, no translation or plateau change.
fig,ax=plt.subplots(3,1,figsize=(7.5,6.8),sharex=True);fig.subplots_adjust(left=.14,right=.97,top=.88,bottom=.11,hspace=.18)
for j,tag in enumerate(['M6','M8','M12','M8lite','M12lite']):
 d=z(RUN/f'high_mach_inventory/local_cpu/validated_results/{tag}/moments.npz',tag+'.npz')
 for a,k in zip(ax,['Rxx_closure','Delta_dvm','S_dvm']):a.plot(d['x_mfp'],d[k],color=C[j],ls='--' if 'lite' in tag else '-',label=tag.replace('lite',' light'));style(a)
for a,l in zip(ax,[r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta_d$',r'$S_d$']):a.set_ylabel(l)
ax[-1].set_xlabel(XH);legend(fig,ax[0],3);save(fig,'high_mach_moments')
# Reviewer sensitivity appendix, fixed archived arrays.
rr=R/'paper_figures/reviewer_revision'; cases=['baseline','velocity_extent','velocity_refine','space_refine','timestep_half','space_extent'];names=['Baseline','Velocity extent','Velocity grid','Spatial grid','Half time step','Spatial extent'];keys=['rho','T','qx','Rxx_closure','Delta_dvm','S_dvm'];grid=np.linspace(-15,15,601);rows=[];zs=[]
def center(x,r,mid):ids=np.where((r[:-1]<mid)&(r[1:]>=mid))[0];assert len(ids)==1;i=ids[0];return x[i]+(mid-r[i])*(x[i+1]-x[i])/(r[i+1]-r[i])
for c in cases:
 d=z(rr/f'inputs/M5_{c}.npz','M5_'+c+'.npz');zs.append(d);x=d['x_mfp'];shift=center(x,d['rho'],(d['rho'][0]+d['rho'][-1])/2);rows.append({k:np.interp(grid+shift,x,d[k]) for k in keys})
diff=np.array([[100*np.linalg.norm(rows[0][k]-r[k])/np.linalg.norm(r[k]) for k in keys] for r in rows[1:]])
tem=pdfplumber.open(IN/'original_A1.pdf').pages[0]
fig,ax=plt.subplots(2,1,figsize=(7.8,6.5),layout='constrained');ax[0].imshow(diff,aspect='auto',cmap='Blues',vmin=0,vmax=max(diff.max(),.55));ax[0].set_xticks(range(6),labs);ax[0].set_yticks(range(5),names[1:]);ax[0].set_title(r'Spatial / velocity sensitivity: $L_2$ difference (%)')
for i in range(5):
 for j in range(6):ax[0].text(j,i,f'{diff[i,j]:.3f}',ha='center',va='center',fontsize=12,color='white' if diff[i,j]>.35 else 'black')
for c,l,col in zip([c for c in tem.curves if len(c['pts'])==6],labs[3:],C):ax[1].plot(range(6),-np.array(c['pts'])[:,1],'o-',label=l,color=col)
ax[1].set_ylim(-203.102424,-38.20048)
ax[1].set_yticks([-127.488596,-90.095971,-52.703347],['0.01','0.02','0.04'])
ax[1].set_xticks(range(6),['Base','Velocity\nextent','Velocity\ngrid','Spatial\ngrid','Half\ntime step','Spatial\nextent']);ax[1].set_ylabel('Last-pair change (%)');style(ax[1]);ax[1].legend(ncol=3,loc='lower right',frameon=False);save(fig,'A1_DVM_sensitivity_summary')
fig,ax=plt.subplots(3,2,figsize=(7.8,7.2));fig.subplots_adjust(top=.83,bottom=.10,left=.13,right=.98,wspace=.37,hspace=.46)
for i,(a,k,l) in enumerate(zip(ax.flat,keys,labs)):
 for j,r in enumerate(rows[1:]):a.plot(grid,100*(r[k]-rows[0][k])/max(abs(rows[0][k])),color=C[j],ls=['-','--','-.',':','-'][j],label=names[j+1])
 a.set_ylabel(l+'\n(% peak baseline)');style(a);a.set_xlabel(XL)
legend(fig,ax[0,0],2);save(fig,'A2_DVM_profile_differences')
# Original DSMC PPC ensembles, density-midpoint aligned with original interpolation.
ppcs=[60,120,240,480,960,1920,3840];ens={}
for ppc in ppcs:
 ds=[]
 for seed in range(1,9):
  d=csv(RUN/f'completed_analysis/ppc_convergence_3840/inputs/ppc{ppc}_seed{seed}/M3_final.csv',f'ppc{ppc}_seed{seed}.csv');x=d['x'];xs=center(x,d['n'],2);g=x[abs(x)<=15];ds.append({k:np.interp(g+xs,x,d[k]) for k in ['QDelta_collision','Rxx_closure','Delta','S','collision_count']})
 ens[ppc]={k:np.array([d[k] for d in ds]) for k in ds[0]}
fig,ax=plt.subplots(1,3,figsize=(8,3.7));fig.subplots_adjust(left=.09,right=.98,bottom=.26,top=.83,wspace=.52)
for i,(a,k,l) in enumerate(zip(ax,['Rxx_closure','Delta','S'],[r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta_c$',r'$S$'])):
 for ppc,col,ls in zip(ppcs[-3:],C,[':','--','-']):
  d=ens[ppc][k];m=d.mean(0);ci=2.3646242516*d.std(0,ddof=1)/np.sqrt(8);a.plot(g,m,color=col,ls=ls,label=f'{ppc} PPC');a.fill_between(g,m-ci,m+ci,color=col,alpha=.16)
 a.set_ylabel(l);a.set_xlabel(XD);style(a)
legend(fig,ax[0],3);save(fig,'Main_DSMC_high_moments')
fig,ax=plt.subplots(3,1,figsize=(7.4,7.3));fig.subplots_adjust(left=.14,right=.98,bottom=.10,top=.86,hspace=.57)
for ppc,col,ls in zip(ppcs[-3:],C,[':','--','-']):
 d=ens[ppc]['QDelta_collision'];m=d.mean(0);ci=2.3646242516*d.std(0,ddof=1)/np.sqrt(8);ax[0].plot(g,m,color=col,ls=ls,label=f'{ppc} PPC');ax[0].fill_between(g,m-ci,m+ci,color=col,alpha=.17);ax[2].plot(g,ens[ppc]['collision_count'].mean(0),color=col,ls=ls)
qd=[100*np.linalg.norm(ens[a]['QDelta_collision'].mean(0)-ens[b]['QDelta_collision'].mean(0))/np.linalg.norm(ens[b]['QDelta_collision'].mean(0)) for a,b in zip(ppcs[:-1],ppcs[1:])];ax[1].plot(ppcs[1:],qd,'o-',color=C[0]);ax[1].set(xscale='log',yscale='log',xlabel='Higher PPC in adjacent pair',ylabel=r'$L_2$ difference (%)');ax[1].set_xticks(ppcs[1:],[str(p) for p in ppcs[1:]]);ax[0].set_ylabel(r'$Q_\Delta^{\rm collision}$');ax[2].set_ylabel('Collisions per cell');ax[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax[0].set_xlabel(XD);ax[2].set_xlabel(XD)
for a in ax:style(a)
legend(fig,ax[0],3);save(fig,'A3_DSMC_collision_source')
# Controlled sparse audit: reuse archived selected hyperparameters, no new search.
sys.path.insert(0,str(Path(__file__).resolve().parent));import sparse_delta_rbf_second_channel as rb
met=json.loads(inp(rr/'metrics.json','reviewer_metrics.json').read_text());d=zs[0];x=d['x_mfp'];x=x-center(x,d['rho'],(d['rho'][0]+d['rho'][-1])/2);delta=rb.plateau_zero(d['Delta_dvm']);rrr=rb.plateau_zero(d['Rxx_closure']);lo,hi=met['sparse_active_interval'];width=hi-lo
fig,ax=plt.subplots(2,1,figsize=(7.5,6.4));fig.subplots_adjust(left=.14,right=.98,top=.85,bottom=.15,hspace=.48);ax[0].plot(x,delta,'k-',label='Reference')
for j,res in enumerate(met['sparse_controlled_audit']):
 xs=np.array(res['xs']);ys=np.interp(xs,x,delta);pred,*_=rb.fit_rbf(x,xs,ys,res['ncenters'],res['width_mult'],res['ridge'],span=(xs[1],xs[-2]));ax[0].plot(x,pred,color=C[j],ls=['--','-.',':'][j],label=res['name']);ax[0].plot(xs,ys,'|',color=C[j]);ax[1].bar(j-.18,res['Delta_error_percent'],.34,color=C[0],label=r'$\Delta_c$' if j==0 else None);ax[1].bar(j+.18,res['R_error_percent'],.34,color=C[1],label=r'$R^{\mathrm{cl}}_{xx}$' if j==0 else None)
ax[0].set(xlim=(lo-.25*width,hi+.25*width),xlabel=XL,ylabel=r'$\Delta_d$');ax[1].set_xticks(range(3),['Active\nuniform','Core gap\n25%','Full-domain\nuniform']);ax[1].set(ylabel='Active-zone error (%)',yscale='log');ax[1].legend(frameon=False)
for a in ax:style(a)
legend(fig,ax[0],2);save(fig,'A4_sparse_core_gap')
# Fixed figures2-4 vector files; no scientific changes.
for n in ['Figure3','Figure4']:
 p=A/(n.lower()+'_large_fonts')/(n+'_large_fonts.pdf');inp(p);shutil.copyfile(IN/p.name,OUT/p.name)
(V/'figure_reproduction/provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Figures generated; inputs',len(meta))
