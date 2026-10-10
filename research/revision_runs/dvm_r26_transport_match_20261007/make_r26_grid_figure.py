from pathlib import Path
import json,hashlib,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
r=Path.cwd();o=r/'output/pdf/shock_comparisons_20261007';b=r/'revision_runs/dvm_r26_transport_match_20261007/completed_analysis'
paths=[r/'revision_runs/r26_m2_pilot_20261007/sweep_20261007/seed/profile.npz',b/'R26_M2_N321_profile.npz',b/'R26_M2_N641_profile.npz']
metrics=json.loads((b/'three_grid_report.json').read_text())['groups']['R26']['metrics'];data=[]
for p in paths:
 a=np.load(p,allow_pickle=False);assert all(np.isfinite(a[k]).all() for k in a.files);mid=(1+16/7)/2;ii=np.flatnonzero((a['rho'][:-1]-mid)*(a['rho'][1:]-mid)<=0);assert len(ii)==1;i=ii[0];xs=a['x'][i]+(mid-a['rho'][i])*(a['x'][i+1]-a['x'][i])/(a['rho'][i+1]-a['rho'][i]);data.append((a['x']-xs,a))
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
f,axs=plt.subplots(3,4,figsize=(15,11));f.subplots_adjust(top=.84,bottom=.09,left=.065,right=.985,hspace=.60,wspace=.33)
f.suptitle('Mach 2 R26: spatial-grid convergence',fontsize=21,fontweight='bold',y=.982)
f.text(.5,.947,'Same domain [-30, 30], equations, transport and far-field states; 161 / 321 / 641 nodes',ha='center',fontsize=11,color='#52616E')
keys=['rho','q','R','Delta','u','sigma','S','theta'];labels=[r'$\rho$',r'$q_x$',r'$R^{cl}_{xx}$',r'$\Delta$',r'$u_x$',r'$\sigma_{xx}$',r'$S=R^{cl}_{xx}+\Delta/3$',r'$T$']
xx=np.linspace(-15,15,1201)
for j,(k,label) in enumerate(zip(keys,labels)):
 ax=axs.flat[j]
 for n,(x,a),col,ls in zip([161,321,641],data,['#E69F00','#D55E00','#0072B2'],[':', '--','-']):ax.plot(x,a[k],color=col,ls=ls,lw=2,label=f'{n} nodes')
 m=metrics['T' if k=='theta' else k];ax.set_title(label+f"\n321 vs 641: {m['medium_fine_percent']:.4f}%",fontsize=12);ax.set_xlim(-15,15);ax.grid(alpha=.23);ax.set_xlabel(r'$(x-x_s)/\ell_1$')
 aa=[np.interp(xx,x,a[k]) for x,a in data];assert abs(100*np.linalg.norm(aa[1]-aa[2])/np.linalg.norm(aa[2])-m['medium_fine_percent'])<1e-10
for j,k in enumerate(['q','R','Delta','S']):
 ax=axs[2,j];m=metrics[k];ax.loglog([.375,.1875],[m['coarse_medium_percent'],m['medium_fine_percent']],'-o',color='#0072B2',lw=2,ms=7);ax.set_title(f"{k}: observed order {m['observed_order']:.2f}");ax.set_xticks([.1875,.375],['0.1875','0.375']);ax.xaxis.set_minor_locator(NullLocator());ax.set_xlabel('Coarser spacing in each pair');ax.set_ylabel('Relative L2 difference (%)',fontsize=10);ax.grid(which='both',alpha=.23)
f.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.918),ncol=3,frameon=False,fontsize=12)
f.text(.5,.035,'Density-midpoint translation only; no width fitting. L2 comparisons on [-15,15], normalized by the 641-node profile.',ha='center',fontsize=10,color='#52616E')
f.text(.5,.012,'All three grids pass the original 1e-6 equation and boundary/phase checks. Observed order is not a certified error bound.',ha='center',fontsize=10,color='#52616E')
for ext in ['png','pdf','svg']:f.savefig(o/('06_M2_R26_three_grids.'+ext),dpi=200,facecolor='white')
(o/'06_M2_R26_three_grids_provenance.json').write_text(json.dumps({'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'metrics':metrics,'alignment':'density midpoint only','norm':'fine grid L2'},indent=2))
print('Eight grid differences independently reproduced; PNG/PDF/SVG saved.')
