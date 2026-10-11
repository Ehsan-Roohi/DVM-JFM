exec((__import__('pathlib').Path(__file__).resolve().parent/'plot_all.py').read_text().split('# Figure1:')[0])
import zipfile
ORIG=Path('C:/Users/roohie/Downloads/Closure_channel_identifiability_and_two_channel_recovery_in_monatomic_kinetic_normal_shocks__22_.zip')
def original(n):
 raw=(IN/('original_'+n+'.pdf')).read_bytes()
 p=IN/('original_'+n+'.pdf');p.write_bytes(raw);meta[p.name]={'original_zip_member':'figures/'+n+'.pdf','sha256':hashlib.sha256(raw).hexdigest(),'method':'Original vector vertices preserved; typography only. Not a raw-data regeneration.'};return pdfplumber.open(p).pages[0]
# Missing final raw plotting contract: preserve original vector traces rather than replace data.
p=original('fig4_sparse_delta_robustness');fig,ax=plt.subplots(3,1,figsize=(7.2,7));fig.subplots_adjust(left=.14,right=.98,bottom=.12,top=.92,hspace=.72)
curves=[c for c in p.curves if len(c['pts'])==4 and c['x0']<50 and c['x1']>160]
for c,col,m,mark in zip(curves,C,[2,3,5],['o','s','^']):
 pts=np.array(c['pts']);x=15+(pts[:,0]-60.326807)/6.0551392;y=(143.633895-pts[:,1])/5.0158148;ax[0].plot(x,y,marker=mark,color=col,label=f'M{m}')
for l,col in zip([l for l in p.lines if l['x0']<37 and l['x1']>169 and isinstance(l['stroking_color'],tuple)],C):ax[0].axhline((143.633895-l['top'])/5.0158148,color=col,lw=1,ls=':')
ax[0].set(xlabel=r'Number of $\Delta$ probes',ylabel=r'$\varepsilon_R$ (%)',title='(a) Sparse-scalar convergence');ax[0].legend(ncol=3,frameon=False)
bars=[r for r in p.rects if r['x0']>200 and r['x1']<328 and r['height']>80 and isinstance(r['non_stroking_color'],tuple)]
for i,r in enumerate(bars):j=i//3;k=i%3;ax[1].bar(k+(j-1)*.23,(143.633895-r['top'])/23.737287,.23,color=C[j],label=['0% noise','0.5% noise','1% noise'][j] if k==0 else None)
ax[1].set_xticks(range(3),['M2','M3','M5']);ax[1].set(ylabel=r'$\varepsilon_R$ (%)',title='(b) Noise robustness: 24 probes',ylim=(0,6.2));ax[1].legend(ncol=3,frameon=False)
bars=[r for r in p.rects if r['x0']>375 and r['x1']<500 and r['height']>1000 and isinstance(r['non_stroking_color'],tuple)]
for i,r in enumerate(bars):j=i//3;k=i%3;y=10**(1+(87.858277-r['top'])/76.633814);ax[2].bar(k+(j-1)*.23,y,.23,color=C[j],label=['M2','M3','M5'][j] if k==0 else None)
ax[2].set_xticks(range(3),[r'$\Delta=0$',r'Complete $\Delta$',r'Sparse $\Delta$']);ax[2].set(yscale='log',ylabel=r'$\varepsilon_R$ (%)',title='(c) Resolving the rank deficiency');ax[2].legend(ncol=3,frameon=False)
ax[2].set_yticks([2,5,10,20,50],['2','5','10','20','50'])
for a in ax:style(a)
save(fig,'fig4_sparse_delta_robustness')
p=original('fig6_dsmc_delta_budget');cs=[c for c in p.curves if len(c['pts'])>100];assert len(cs)==7
fig,ax=plt.subplots(2,2,figsize=(7.8,7.1));fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.86,hspace=.95,wspace=.36)
bounds=[(74.900617,363.600456,22.324999832,206.483541832),(446.864689,735.564529,22.324999832,206.483541832),(74.900617,363.600456,288.724999832,472.883541832),(446.864689,735.564529,288.724999832,472.883541832)]
cal=[(114.404270832,83.708428),(159.745363832,3.374185),(346.041627832,1.4255138),(346.041627832,118.47107)]
counts=[2,1,2,2];labels=[['Heat flux','Normal stress'],['Scalar excess'],['Collision production','Spatial budget'],['Collision production',r'$-\Delta$']];titles=['(a) Shock observables','(b) Scalar excess','(c) Scalar production','(d) Normalised profiles'];ylabs=['Normalised value',r'$\Delta_c$',r'$Q_\Delta$','Normalised value']
# Vector coordinates retain exactly the old plot-space geometry; ticks retain original scale.
q=0
for i,a in enumerate(ax.flat):
 x0,x1,top,bottom=bounds[i]
 for j in range(counts[i]):
  c=cs[q];q+=1;pts=np.array(c['pts']);x=-60+(pts[:,0]-x0)*120/(x1-x0);zero,scale=cal[i];y=(zero-pts[:,1])/scale;a.plot(x,y,color=['black',C[1] if i!=3 else C[2]][j],ls='-' if j==0 else '--',label=labels[i][j])
 zero,scale=cal[i];a.set(xlim=(-60,60),ylim=((zero-bottom)/scale,(zero-top)/scale),xlabel=r'$x/\lambda_1$',ylabel=ylabs[i],title=titles[i]);style(a)
 if i!=1:a.legend(loc='upper center',bbox_to_anchor=(.5,1.54),ncol=1,frameon=False,fontsize=11)
ax[1,0].text(.03,.06,r'$\varepsilon_Q=0.170$'+'\n'+r'$\rho_Q=0.987$',transform=ax[1,0].transAxes,fontsize=11)
ax[1,1].text(.03,.06,r'$\rho(Q_\Delta,-\Delta)=0.968$',transform=ax[1,1].transAxes,fontsize=11)
save(fig,'fig6_dsmc_delta_budget')
# M2 and M3 corrected raw source files in v22, not the obsolete previous R26 curve.
FIELDS=['rho','T','q','R','Delta','S'];LAB=[r'$\rho$',r'$T$',r'$q_x$',r'$R^{\mathrm{cl}}_{xx}$',r'$\Delta_c$',r'$S$']
def load(p,name):
 d=dict(z(p,name));
 if 'fields' in d:d.update({str(k):d['fields'][:,i] for i,k in enumerate(d['columns'])})
 if 'theta' in d:d['T']=d['theta']
 return d
def midpoint(d,m):
 y=d['rho']-(1+4*m*m/(m*m+3))/2;ids=np.flatnonzero(y[:-1]*y[1:]<=0);i=ids[np.argmin(abs(d['x'][ids]))];return d['x'][i]-y[i]*np.diff(d['x'])[i]/(y[i+1]-y[i])
def models(m):
 if m in [2,3]:
  b=V/f'review/M{m}_raw';return [(load(b/f'DSMC_2000ppc_M{m}.npz',f'raw_DSMC_M{m}.npz'),'DSMC 2000 PPC, raw','gray','-'),(load(b/f'DVM_Shakhov_M{m}.npz',f'raw_DVM_M{m}.npz'),'Shakhov DVM snapshot',C[0],'--'),(load(b/f'R26_accepted_Shakhov_M{m}.npz',f'raw_R26_M{m}.npz'),'Corrected Shakhov R26',C[1],'-')]
 return [(load(RUN/f'bird_dsmc_m2_m5_high_particles_20261009/collected/M{m}/profiles.npz',f'raw_DSMC_M{m}.npz'),'DSMC 2000 PPC, raw','gray','-'),(load(RUN/f'dvm_m3_m5_transport_match_20261008/live_profiles_20261009T212610Z/M{m}_x{m*640}.npz',f'raw_DVM_M{m}.npz'),'Shakhov DVM snapshot',C[0],'--')]
fig,ax=plt.subplots(3,2,figsize=(7.7,7));fig.subplots_adjust(top=.82,bottom=.10,left=.12,right=.98,wspace=.40,hspace=.42)
for a,k,l in zip(ax.flat,FIELDS,LAB):
 for d,n,col,ls in models(2):a.plot(d['x']-midpoint(d,2),d[k],color=col,ls=ls,label=n,lw=1 if 'DSMC' in n else 1.8)
 a.set(xlim=(-15,15),xlabel=r'$(x-x_{1/2})/\lambda_1$',ylabel=l);style(a)
legend(fig,ax[0,0],1);save(fig,'M2_DSMC2000_cross_method')
# Figure15 rearranges 18 panels into six rows of three (one Mach per column).
def smooth(x,y):
 n=max(5,int(round(.75/np.median(np.diff(x)))));n+=1-n%2;r=n//2;out=np.empty(len(y))
 for i in range(len(y)):
  lo=max(0,i-r);hi=min(len(y),i+r+1);t=(x[lo:hi]-x[i])/max(x[hi-1]-x[lo],np.median(np.diff(x)));out[i]=np.linalg.lstsq(np.c_[np.ones(len(t)),t,t*t],y[lo:hi],rcond=None)[0][0]
 return out
fig,ax=plt.subplots(6,3,figsize=(8,10.4));fig.subplots_adjust(top=.91,bottom=.08,left=.10,right=.98,wspace=.42,hspace=.40)
for j,m in enumerate([3,4,5]):
 ds=models(m)
 for i,(k,l) in enumerate(zip(FIELDS,LAB)):
  a=ax[i,j]
  for d,n,col,ls in ds:
   xc=d['x']-midpoint(d,m);a.plot(xc,d[k],color=col,ls=ls,lw=.6 if 'DSMC' in n else 1.5,alpha=.65 if 'DSMC' in n else 1,label=n)
   if 'DSMC' in n:a.plot(xc,smooth(xc,d[k]),color=C[2],label='DSMC visual guide',lw=1.7)
  a.set_xlim(-15,15);style(a)
  if j==0:a.set_ylabel(l)
  if i==0:a.set_title(rf'$M_1={m}$')
  if i==5:a.set_xlabel(r'$(x-x_{1/2})/\lambda_1$')
legend(fig,ax[0,0],2);save(fig,'M3_M5_DSMC2000_DVM_R26')
# Mach8: saved comparison vertices, normalized jumps and same shared density shift.
b=RUN/'m8_reference/paper_comparison';fig,ax=plt.subplots(2,2,figsize=(7.8,5.5),sharex=True,gridspec_kw={'height_ratios':[2,1]});fig.subplots_adjust(top=.79,bottom=.13,left=.12,right=.98,wspace=.35,hspace=.20)
for j,k in enumerate(['rho','T']):
 for i,name in enumerate(['baseline','time_extend','space_refine']):
  d=csv(b/f'{name}_{k}_comparison.csv',f'm8_{name}_{k}.csv');ax[0,j].plot(d['x_lambda1'],d['ours_aligned'],color=C[i],ls=['-',':','--'][i],label=['Baseline','Extended time','Spatial refinement'][i]);ax[1,j].plot(d['x_lambda1'],100*d['difference'],color=C[i],ls=['-',':','--'][i])
 ax[0,j].plot(d['x_lambda1'],d['paper_vector'],'k-',label='Chen et al., S-model');ax[0,j].set_title('Density' if k=='rho' else 'Temperature');ax[0,j].set_ylabel('Normalised jump');ax[1,j].set(xlabel=r'$x/\lambda_1$',ylabel='Difference (pp)')
 for a in ax[:,j]:style(a);a.set_xlim(-20,15)
legend(fig,ax[0,0],2);save(fig,'m8_paper_comparison')
(V/'figure_reproduction/provenance_remaining.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Remaining figures completed')
