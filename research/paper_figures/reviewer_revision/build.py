from pathlib import Path
import json, hashlib, sys, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
ROOT=R.parents[1]; RUN=ROOT/'revision_runs'
sys.path.insert(0,str(RUN/'source'))
import sparse_delta_rbf_second_channel as rb
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':10,'lines.linewidth':1.5,'svg.fonttype':'none','pdf.fonttype':42})
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#222222']
prov={}; metrics={}
def record(p):
    prov[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
def save(f,n):
    for ext in ['pdf','svg','png']: f.savefig(R/(n+'.'+ext),dpi=320,bbox_inches='tight')
    plt.close(f)
def center(x,r,mid):
    j=np.where((r[:-1]<mid)&(r[1:]>=mid))[0]; assert len(j)==1
    i=j[0];return x[i]+(mid-r[i])*(x[i+1]-x[i])/(r[i+1]-r[i])
def rel(a,b):return float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
def panel(ax,i):
    ax.text(.02,.96,'('+chr(97+i)+')',va='top',transform=ax.transAxes)
    ax.grid(alpha=.16)
cases=['M5_baseline','M5_velocity_extent','M5_velocity_refine','M5_space_refine','M5_timestep_half','M5_space_extent']
names=['Baseline','Velocity extent','Velocity grid*','Spatial grid','Half time step','Spatial extent']
ks=['rho','T','qx','Rxx_closure','Delta_dvm','S_dvm']
labs=[r'$\rho$',r'$T$',r'$q_x$',r'$R$',r'$\Delta_{\rm DVM}$',r'$S_{\rm DVM}$']
grid=np.linspace(-15,15,601); rows=[]; zs=[]
for case in cases:
    p=R/'inputs'/f'{case}.npz';record(p);z=np.load(p);zs.append(z)
    x=z['x_mfp'];shift=center(x,z['rho'],.5*(z['rho'][0]+z['rho'][-1]))
    rows.append({k:np.interp(grid+shift,x,z[k]) for k in ks})
diff=np.array([[rel(rows[0][k],row[k]) for k in ks] for row in rows[1:]])
metrics['DVM_relative_L2_percent']={n:dict(zip(ks,v.tolist())) for n,v in zip(names[1:],diff)}
p=RUN/'convergence_review_20261002.jsonl';record(p)
tem=[json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
f,ax=plt.subplots(1,2,figsize=(8.2,3.6),layout='constrained')
im=ax[0].imshow(diff,aspect='auto',cmap='Blues',vmin=0,vmax=max(diff.max(),.55))
ax[0].set_xticks(range(6),labs);ax[0].set_yticks(range(5),names[1:]);ax[0].set_title('DVM sensitivity: relative $L_2$ (%)')
for i in range(5):
    for j in range(6): ax[0].text(j,i,f'{diff[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if diff[i,j]>.35 else 'black')
for k,lab,col in zip(ks[3:],labs[3:],colors):
    ax[1].plot(range(6),[q['late_temporal_drift_percent'][k] for q in tem],'o-',label=lab,c=col)
ax[1].set_xticks(range(6),names,rotation=40,ha='right');ax[1].set_ylabel('Last saved-pair change (%)');ax[1].set_yscale('log');ax[1].legend(frameon=False);ax[1].grid(alpha=.2)
ax[1].set_title('Late temporal change (unequal intervals)')
f.suptitle('Mach 5 BGK DVM: *velocity grid at 50,000 / 52,000 steps',fontsize=10)
save(f,'A1_DVM_sensitivity_summary')
f,axs=plt.subplots(2,3,figsize=(8,5.2),layout='constrained')
for i,(ax,k,lab) in enumerate(zip(axs.flat,ks,labs)):
    for j,row in enumerate(rows[1:]):
        ax.plot(grid,100*(row[k]-rows[0][k])/np.max(np.abs(rows[0][k])),c=colors[j],ls=['-','--','-.',':','-'][j],label=names[j+1])
    ax.set_xlabel(r'$(x-x_s)/\lambda_1$');ax.set_ylabel(lab+' difference (% peak baseline)');panel(ax,i)
axs[0,0].legend(fontsize=7,frameon=False,loc='lower right')
save(f,'A2_DVM_profile_differences')
# Every seed is aligned independently, without smoothing or extrapolation.
ppcs=[60,120,240,480,960,1920,3840]; ensembles={}; raw={}
for ppc in ppcs:
    data=[]
    for seed in range(1,9):
        p=RUN/f'completed_analysis/ppc_convergence_3840/inputs/ppc{ppc}_seed{seed}/M3_final.csv';record(p)
        z=np.genfromtxt(p,delimiter=',',names=True);x=z['x'];shift=center(x,z['n'],2.)
        g=x[np.abs(x)<=15]
        assert g[0]+shift>=x[0] and g[-1]+shift<=x[-1]
        data.append({k:np.interp(g+shift,x,z[k]) for k in ['QDelta_collision','Rxx_closure','Delta','S','collision_count']})
    ensembles[ppc]={k:np.array([d[k] for d in data]) for k in data[0]}
f,axs=plt.subplots(1,3,figsize=(9.1,3.1),layout='constrained')
for ppc,col,ls in zip(ppcs[-3:],colors,[':','--','-']):
    a=ensembles[ppc]['QDelta_collision'];m=a.mean(0);ci=2.3646242516*a.std(0,ddof=1)/np.sqrt(8)
    axs[0].plot(g,m,c=col,ls=ls,label=str(ppc));axs[0].fill_between(g,m-ci,m+ci,color=col,alpha=.17)
    axs[2].plot(g,ensembles[ppc]['collision_count'].mean(0),c=col,ls=ls)
qdiff=[rel(ensembles[a]['QDelta_collision'].mean(0),ensembles[b]['QDelta_collision'].mean(0)) for a,b in zip(ppcs[:-1],ppcs[1:])]
metrics['DSMC_Qcollision_adjacent_L2_percent']=dict(zip([f'{a}-{b}' for a,b in zip(ppcs[:-1],ppcs[1:])],qdiff))
axs[1].plot(ppcs[1:],qdiff,'o-',c=colors[0]);axs[1].set_xscale('log',base=2);axs[1].set_yscale('log');axs[1].set_xticks(ppcs[1:],[str(p) for p in ppcs[1:]],rotation=40)
axs[1].set_xlabel('Higher PPC in adjacent pair');axs[1].set_ylabel(r'$Q_\Delta$: relative $L_2$ difference (%)')
axs[0].set_ylabel(r'$Q_\Delta^{\rm collision}$');axs[0].legend(title='PPC; 8 seeds',fontsize=7,frameon=False)
axs[2].set_ylabel('Accepted collisions per cell');axs[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
for i,ax in enumerate(axs):
    if i!=1: ax.set_xlabel(r'$(x-x_s)/\lambda_1$')
    panel(ax,i)
save(f,'A3_DSMC_collision_source')
f,axs=plt.subplots(1,3,figsize=(8.5,3),layout='constrained')
for i,(k,lab) in enumerate(zip(['Rxx_closure','Delta','S'],[r'$R$',r'$\Delta$',r'$S$'])):
    for ppc,col,ls in zip(ppcs[-3:],colors,[':','--','-']):
        a=ensembles[ppc][k];m=a.mean(0);ci=2.3646242516*a.std(0,ddof=1)/np.sqrt(8)
        axs[i].plot(g,m,c=col,ls=ls,label=str(ppc));axs[i].fill_between(g,m-ci,m+ci,color=col,alpha=.16)
    axs[i].set_xlabel(r'$(x-x_s)/\lambda_1$');axs[i].set_ylabel(lab);panel(axs[i],i)
axs[0].legend(title='PPC',fontsize=7,frameon=False)
save(f,'Main_DSMC_high_moments')
# Controlled sampling audit: exact combined channel isolates interpolation error.
# Full-reference active-zone selection is an oracle diagnostic, not blind recovery.
record(RUN/'source/sparse_delta_rbf_second_channel.py')
z=zs[0];x=z['x_mfp'];shift=center(x,z['rho'],.5*(z['rho'][0]+z['rho'][-1]));x=x-shift
delta=rb.plateau_zero(z['Delta_dvm']);rr=rb.plateau_zero(z['Rxx_closure']);s=rr+delta/3
mask=rb.active_mask(delta,.02);lo,hi=x[mask].min(),x[mask].max();width=hi-lo
aa=max(x[0],lo-.25*width);bb=min(x[-1],hi+.25*width)
peak=x[np.argmax(np.abs(delta))];results=[]
f,axs=plt.subplots(1,2,figsize=(8,3.4),layout='constrained')
axs[0].plot(x,delta,c='black',lw=2,label='Reference')
for j,(name,gap) in enumerate([('Active uniform',0),('Core gap 25%',.25),('Full-domain uniform',None)]):
    if gap is None: xs=rb.select_sensors(x,delta,24,'all')
    else:
        if gap==0: inner=np.linspace(aa,bb,24)
        else:
            left=peak-gap*width/2;right=peak+gap*width/2
            # 24 probes distributed over the same span with a deliberate central gap.
            t=np.linspace(0,(left-aa)+(bb-right),24)
            inner=np.where(t<=left-aa,aa+t,right+t-(left-aa))
        xs=np.unique(np.r_[x[0],inner,x[-1]])
    ys=np.interp(xs,x,delta);span=(float(xs[1]),float(xs[-2]));best=None
    for nc in [28,32,48]:
        for wm in [.5,.6,.8,1.,1.2,1.5]:
            for ridge in [1e-10,1e-8,1e-6,1e-4,1e-2]:
                item=(rb.loo_score(xs,ys,nc,wm,ridge,span),nc,wm,ridge)
                if best is None or item<best:best=item
    pred,*_=rb.fit_rbf(x,xs,ys,best[1],best[2],best[3],span=span)
    rpred=s-pred/3
    e=rel(rpred[mask],rr[mask]);ed=rel(pred[mask],delta[mask])
    results.append(dict(name=name,sensors=len(xs),core_sensors=int(((xs>=lo)&(xs<=hi)).sum()),R_error_percent=e,Delta_error_percent=ed,loo=best[0],ncenters=best[1],width_mult=best[2],ridge=best[3],xs=xs.tolist()))
    axs[0].plot(x,pred,c=colors[j],ls=['--','-.',':'][j],label=name)
    axs[0].plot(xs,ys,'|',c=colors[j],ms=6)
    axs[1].bar(j-.17,ed,width=.32,color=colors[0],label=r'$\Delta$' if j==0 else None)
    axs[1].bar(j+.17,e,width=.32,color=colors[1],label=r'$R$' if j==0 else None)
    print('SPARSE',name,e,ed,flush=True)
axs[0].set_xlim(aa,bb);axs[0].set_xlabel(r'$(x-x_s)/\lambda_1$');axs[0].set_ylabel(r'$\Delta_{\rm DVM}$');axs[0].legend(fontsize=7,frameon=False)
axs[1].set_xticks(range(3),[q['name'] for q in results],rotation=20,ha='right');axs[1].set_ylabel('Active-zone relative $L_2$ error (%)');axs[1].set_yscale('log');axs[1].legend(frameon=False)
for i,ax in enumerate(axs):panel(ax,i)
save(f,'A4_sparse_core_gap')
metrics['sparse_controlled_audit']=results
metrics['sparse_active_interval']=[float(lo),float(hi)]
metrics['sparse_reference_channel']='exact R + Delta/3; not a budget-reconstruction validation'
(R/'metrics.json').write_text(json.dumps(metrics,indent=2))
record(Path(__file__))
(R/'provenance.json').write_text(json.dumps(prov,indent=2))
print(json.dumps(metrics['DSMC_Qcollision_adjacent_L2_percent']),flush=True)
