from pathlib import Path
import json, hashlib, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
base=R.parent/'revision_runs'
inputs=base/'completed_analysis/ppc_convergence_3840/inputs'
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':10,'axes.linewidth':.7,'lines.linewidth':1.6,'svg.fonttype':'none','savefig.facecolor':'white'})
colors=['#0072B2','#D55E00','#009E73']
styles=['--','-.','-']
provenance={}
def record(p):provenance[p.relative_to(R.parent).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,name):
    fig.savefig(R/(name+'.svg'),bbox_inches='tight')
    fig.savefig(R/(name+'.png'),dpi=400,bbox_inches='tight')
    plt.close(fig)
def panel(ax,i,label):
    ax.text(.03,.08,'('+chr(97+i)+')',transform=ax.transAxes,va='bottom')
    ax.set_ylabel(label);ax.set_xlabel(r'$x-x_s$');ax.grid(alpha=.16)
def center(x,n,mid):
    ii=np.where((n[:-1]<mid)&(n[1:]>=mid))[0]
    assert len(ii)==1
    j=ii[0];return x[j]+(mid-n[j])*(x[j+1]-x[j])/(n[j+1]-n[j])
fields=['n','T','qx','Rxx_closure','Delta','S']
labels=[r'$n$',r'$T$',r'$q_x$',r'$R$',r'$\Delta$',r'$S$']
fig,axs=plt.subplots(2,3,figsize=(7.2,4.7),layout='constrained')
for ppc,col,ls in zip([960,1920,3840],colors,styles):
    sets={k:[] for k in fields}
    for seed in range(1,9):
        p=inputs/f'ppc{ppc}_seed{seed}'/'M3_final.csv';record(p)
        z=np.genfromtxt(p,delimiter=',',names=True);x=z['x'];mask=abs(x)<=15
        shift=center(x,z['n'],2.)
        for k in fields:sets[k].append(np.interp(x[mask]+shift,x,z[k]))
    for ax,k in zip(axs.flat,fields):
        a=np.array(sets[k]);m=a.mean(0);ci=2.364624251592784*a.std(0,ddof=1)/np.sqrt(8)
        ax.plot(x[mask],m,c=col,ls=ls,label=str(ppc));ax.fill_between(x[mask],m-ci,m+ci,color=col,alpha=.13)
for i,(ax,lab) in enumerate(zip(axs.flat,labels)):panel(ax,i,lab)
axs[0,0].legend(title='PPC',fontsize=8,frameon=False,loc='lower right')
save(fig,'Fig1_DSMC_high_PPC')

dp=base/'completed_analysis/ppc_convergence_3840/diagnostics.json';record(dp);d=json.loads(dp.read_text())
fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
hi=[120,240,480,960,1920,3840];lo=[60,120,240,480,960,1920]
for k,lab,col in zip(['Rxx_closure','Delta','S'],labels[3:],colors):
    y=[d['adjacent'][f'{a}-{b}'][k]['L2_percent'] for a,b in zip(lo,hi)]
    axs[0].plot(hi,y,'o-',color=col,label=lab,ms=4)
    axs[1].plot(lo,[d['vs3840'][str(p)][k] for p in lo],'o-',color=col,ms=4)
for i,ax in enumerate(axs):
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.grid(alpha=.2);ax.set_ylabel(r'Relative $L_2$ difference (%)');ax.text(.03,.08,'('+chr(97+i)+')',transform=ax.transAxes,va='bottom')
    ticks=hi if i==0 else lo;ax.set_xticks(ticks,[str(p) for p in ticks],rotation=35);ax.set_xlabel('Higher PPC in adjacent pair' if i==0 else 'PPC (reference: 3840)')
axs[0].legend(frameon=False)
save(fig,'Fig2_DSMC_particle_sensitivity')

p=base/'steady_check/dvm_review_plots/inputs/M5_baseline/result.npz';shutil.copy2(p,R/'inputs/dvm_baseline.npz')
zs=[]
for name in ['dvm_baseline','dvm_space_refine']:
    p=R/'inputs'/f'{name}.npz';record(p);zs.append(np.load(p))
assert provenance['paper_figures/inputs/dvm_space_refine.npz']=='7ca2d55e51fef84a93b01af307cb4daf8f5b82f93f07802c5a52fffe60e6af60'
ks=['rho','T','qx','Rxx_closure','Delta_dvm','S_dvm'];lab=[r'$\rho$',r'$T$',r'$q_x$',r'$R$',r'$\Delta_{\rm DVM}$',r'$S_{\rm DVM}$']
grid=np.linspace(-15,15,601);curves=[]
fig,axs=plt.subplots(2,3,figsize=(7.2,4.7),layout='constrained')
for z,n,col,ls in zip(zs,[2200,3299],colors,styles):
    x=z['x_mfp'];mid=.5*(z['rho'][0]+z['rho'][-1]);shift=center(x,z['rho'],mid)
    row={k:np.interp(grid+shift,x,z[k]) for k in ks};curves.append(row)
    for ax,k in zip(axs.flat,ks):ax.plot(grid,row[k],c=col,ls=ls,label=rf'$N_x={n}$')
for i,(ax,l) in enumerate(zip(axs.flat,lab)):
    panel(ax,i,l);ax.set_xlabel(r'$(x-x_s)/\lambda_1$')
axs[0,0].legend(frameon=False,fontsize=8,loc='lower right')
save(fig,'Fig3_DVM_spatial_sensitivity')
metrics={k:float(100*np.linalg.norm(curves[0][k]-curves[1][k])/np.linalg.norm(curves[1][k])) for k in ks}
(R/'metrics.json').write_text(json.dumps({'DVM_aligned_relative_L2_percent':metrics,'DSMC_1920_3840':d['adjacent']['1920-3840']},indent=2))
(R/'provenance.json').write_text(json.dumps(provenance,indent=2))
(R/'captions_EN.txt').write_text('''Figure 1. Particle-number sensitivity of the Mach-3 hard-sphere DSMC shock. Curves show eight-seed means for PPC=960,1920,3840; shaded bands are pointwise 95% Student-t intervals across seeds (df=7), excluding systematic error. Each time-averaged profile is shifted to its n=2 crossing, using linear interpolation without smoothing. Alignment does not remove temporal shock wandering. Variables retain the solver nondimensionalization. These original-solver results retain the identified pair-selection bias and are provisional pending corrected-solver assessment.

Figure 2. Relative L2 differences of aligned ensemble-mean high moments over |x-xs|<=15: (a) adjacent PPC levels, normalized by the higher-PPC profile; (b) comparison with PPC3840, a finite-particle reference, not an exact solution. Curves measure sensitivity, not exact-solution error or statistical significance. Eight seeds per level. Seed uncertainty and systematic pair-selection bias prevent interpreting small differences alone as proof of convergence.

Figure 3. Spatial-resolution sensitivity of the Mach-5 BGK DVM shock at Nx=2200 and3299, after52000 and78000 steps respectively, with approximately matched physical integration times and the same141x27x27 velocity grid. Profiles are aligned by their density midpoint. Refinement also changes the CFL-controlled time step; this is a coupled spatial/time-discretization comparison. Delta_DVM uses the discrete-Maxwellian reference and is not the raw DSMC fourth-moment excess. Two grids do not establish asymptotic convergence or velocity-grid independence. All final fields and source/checkpoint provenance have been validated.
''',encoding='utf-8')
print(json.dumps(metrics))


