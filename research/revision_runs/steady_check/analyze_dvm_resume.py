import json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/'dvm_review_plots'
cases=json.loads((R.parents[1]/'cases.json').read_text())
fields=['rho','ux','T','qx','sigma_xx','Rxx_closure','Delta_dvm','S_dvm']
labels=[r'$\rho$',r'$u_x$',r'$T$',r'$q_x$',r'$\sigma_{xx}$',r'$R_{xx}$',r'$\Delta$',r'$S$']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
report={'method':'Saved physical coordinates, no alignment or smoothing. Relative Linf =100*max|new-old|/max|new| over full domain. 0.1% over each of last three stored intervals is an illustrative diagnostic, not a prespecified solver stopping or scientific acceptance test. Intervals have different physical durations across cases.','cases':{}}
fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
for c in cases:
    d=R/'inputs'/c['name']; paths=sorted((d/'history').glob('step*.npz'))
    settings=json.loads((d/'result.npz.settings.json').read_text())
    hist=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as z:
            data={k:z[k].astype(float) for k in fields+['x_mfp']}
        assert all(np.isfinite(v).all() for v in data.values())
        hist.append((int(p.stem[4:]),data))
    dx=float(np.float32(hist[0][1]['x_mfp'][1])-np.float32(hist[0][1]['x_mfp'][0]));dt=c['cfl']*dx/c['vmax']
    changes=[]
    for (a,za),(b,zb) in zip(hist[:-1],hist[1:]):
        assert np.array_equal(za['x_mfp'],zb['x_mfp'])
        changes.append({'from':a,'to':b,'time':b*dt,'interval_time':(b-a)*dt,'percent':{k:float(100*np.max(np.abs(zb[k]-za[k]))/max(np.max(np.abs(zb[k])),1e-30)) for k in fields}})
    stable=len(changes)>=3 and all(max(h['percent'].values())<.1 for h in changes[-3:])
    done=(d/'DONE.json').exists()
    report['cases'][c['name']]={'done':done,'latest_step':hist[-1][0],'target_steps':c['steps'],'dt':dt,'finite':True,'small_drift_last_three_intervals':stable,'latest_three_changes':changes[-3:],'all_changes':changes}
    short=c['name'].replace('M5_','').replace('_',' ')
    for ax,keys in zip(axes,[fields[:5],fields[5:]]):
        ax.semilogy([h['time'] for h in changes],[max(h['percent'][k] for k in keys) for h in changes],label=short)
    if stable or done:
        f,aa=plt.subplots(2,4,figsize=(14,7),layout='constrained')
        selected=hist[-4:]
        for j,(step,z) in enumerate(selected):
            for ax,k,label in zip(aa.flat,fields,labels):
                ax.plot(z['x_mfp'],z[k],lw=1.6,label=f'step {step:,} (t={step*dt:.2f})',alpha=.8)
                ax.set_xlim(-18,18);ax.set_xlabel(r'$x/\lambda_1$');ax.set_ylabel(label);ax.grid(alpha=.18)
        aa[0,0].legend(fontsize=8)
        title='Completed baseline: late-time profiles' if done else short.title()+': provisional late-time profiles (unfinished)'
        f.suptitle(title+'\nMach 5 BGK DVM; temporal stability does not establish grid independence',fontsize=12)
        for ext in ['png','pdf']:f.savefig(R/(c['name']+'_profiles.'+ext))
        plt.close(f)
for ax,title in zip(axes,['Largest change among density, velocity, T, heat flux, stress','Largest change among R, Delta, S']):
    ax.axhline(.1,color='k',ls='--',lw=.9,label='0.1% diagnostic guide')
    ax.set_xlabel('Physical simulation time');ax.set_ylabel('Change over previous saved interval (%)');ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
axes[1].legend(fontsize=8,ncol=2)
fig.suptitle('DVM temporal drift: all six cases\nEach curve compares successive snapshots on its own grid; saved interval duration differs',fontsize=12)
for ext in ['png','pdf']:fig.savefig(R/('temporal_drift.'+ext))
plt.close(fig)
(R/'diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps({n:{k:v for k,v in d.items() if k!='all_changes'} for n,d in report['cases'].items()},indent=2))
