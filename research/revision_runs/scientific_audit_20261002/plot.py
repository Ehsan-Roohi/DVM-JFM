from pathlib import Path
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent
load=lambda n:json.loads((P/(n+'.json')).read_text())
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for name,d in load('distribution_entropy')['cases'].items():
 if 'rows' not in d:continue
 rows=sorted(d['rows'],key=lambda r:r['x_shifted']);x=np.array([r['x_shifted'] for r in rows]);mask=abs(x)<=15
 label=name.replace('M5_','').replace('_',' ')
 axes[0,0].plot(x[mask],[r['entropy_order4']['entropy_direct_over_rho'] for r,m in zip(rows,mask) if m],'.-',label=label)
 axes[0,1].semilogy(x[mask],[r['tail_c4_fractions']['0.9'] for r,m in zip(rows,mask) if m],'.-',label=label)
axes[0,0].set(xlabel='Aligned position',ylabel='Local entropy production / density',title='(a) ES-FP operator on BGK distributions')
axes[0,0].legend(fontsize=8)
axes[0,1].set(xlabel='Aligned position',ylabel='Resolved outer-box c4 fraction',title='(b) Outer 10% of velocity box')
keys=['Rxx_closure','Delta','S','QDelta_collision'];labels=['R','Delta','S','Collision source'];x=np.arange(4)
for offset,ppc in [(-.18,'120'),(.18,'960')]:axes[1,0].bar(x+offset,[load('paired_sampler')['results'][ppc][k]['mean_difference_L2_percent'] for k in keys],width=.36,label='PPC '+ppc)
axes[1,0].set(xticks=x,xticklabels=labels,ylabel='Relative L2 difference (%)',title='(c) Original vs corrected: two seeds')
axes[1,0].legend()
axes[1,1].bar(x,[load('time_statistics')['aggregate'][k]['early_late_ensemble_L2_percent'] for k in keys],color='#0072B2')
axes[1,1].set(xticks=x,xticklabels=labels,ylabel='Relative L2 difference (%)',title='(d) Early vs late: eight original seeds')
fig.savefig(P/'scientific_diagnostics.png',dpi=200);fig.savefig(P/'scientific_diagnostics.svg')
plt.close(fig)
print('Diagnostic plots saved; not a convergence certificate.')
