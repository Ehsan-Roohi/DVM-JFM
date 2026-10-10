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
names=['Baseline','Velocity extent','Velocity grid','Spatial grid','Half time step','Spatial extent']
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
f.suptitle('Mach 5 BGK DVM: all prescribed integrations complete',fontsize=10)
save(f,'A1_DVM_sensitivity_summary')
f,axs=plt.subplots(2,3,figsize=(8,5.2),layout='constrained')
for i,(ax,k,lab) in enumerate(zip(axs.flat,ks,labs)):
    for j,row in enumerate(rows[1:]):
        ax.plot(grid,100*(row[k]-rows[0][k])/np.max(np.abs(rows[0][k])),c=colors[j],ls=['-','--','-.',':','-'][j],label=names[j+1])
    ax.set_xlabel(r'$(x-x_s)/\lambda_1$');ax.set_ylabel(lab+' difference (% peak baseline)');panel(ax,i)
axs[0,0].legend(fontsize=7,frameon=False,loc='lower right')
save(f,'A2_DVM_profile_differences')

(R/"final_DVM_metrics.json").write_text(json.dumps(metrics,indent=2))
