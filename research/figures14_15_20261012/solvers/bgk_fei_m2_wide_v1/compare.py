"""Evaluate completed runs; never substitute transient outputs for steady curves."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def centered(x,rho):
    mid=(1+16/7)/2; y=rho-mid
    ids=np.flatnonzero(y[:-1]*y[1:]<=0)
    if len(ids)!=1: raise ValueError('Density midpoint is not unique')
    i=ids[0]; xs=x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])
    return x-xs,float(xs)
def difference(x,y,xr,yr):
    xx=np.linspace(-15,15,1201)
    if min(x[-1],xr[-1])<15 or max(x[0],xr[0])>-15:raise ValueError('Incomplete common window')
    a=np.interp(xx,x,y);b=np.interp(xx,xr,yr);d=a-b
    return {'max_percent_density_jump':float(100*np.max(abs(d))/(16/7-1)),
            'rms_percent_density_jump':float(100*np.sqrt(np.mean(d*d))/(16/7-1)),
            'relative_L2_percent_reference_density':float(100*np.linalg.norm(d)/np.linalg.norm(b))}
def main():
    manifest=json.loads((P/'manifest.json').read_text()); curves={};states={};missing=[]
    import bgk_reference as original
    import shakhov_reference as ref
    original_checks={}
    for c in manifest['cases']:
        out=P/'runs'/c['name']; st=out/'STATUS.json'
        if not st.exists(): missing.append(c['name']);continue
        s=json.loads(st.read_text()); states[c['name']]=s
        if s['config']!=c:raise ValueError('Configuration mismatch')
        for n,h in s['sha256'].items():
            if sha(out/n)!=h:raise ValueError('Closed payload changed')
        if not s['stationary'] or s['final_rhs']>=1e-7 or s['stable']<5:
            missing.append(c['name']);continue
        ck=np.load(out/'checkpoint.npz',allow_pickle=False);f=ck['f']
        xx,v,w,unused,left,right=ref.setup(c)
        aa=ref.moments(f,v,w);bb=ref.moments(original.advance(f,v,w,left,right,c,s['dt']),v,w)
        original_rhs=float(np.max(np.max(abs(bb-aa),axis=0)/(s['dt']*np.maximum(np.max(abs(aa),axis=0),1e-8))))
        original_checks[c['name']]=original_rhs
        if original_rhs>=1e-7:missing.append(c['name']+' original-RHS');continue
        z=np.load(out/'profiles.npz',allow_pickle=False);rho=z['fields'][:,0]
        x,shift=centered(z['x'],rho); curves[c['name']]=(x,rho,shift)
    if missing:
        (P/'COMPARISON_NOT_QUALIFIED.json').write_text(json.dumps({'state':'NOT_ALL_STATIONARY','cases':missing,'actual_status':states,'no_grid_accuracy_or_model_error_claim':True},indent=2))
        print('NOT_ALL_STATIONARY: '+', '.join(missing));return 2
    base=curves['M2_domain']; grid={}
    for n in ('M2_space','M2_velocity','M2_halfdt','M2_domain90'):
        grid[n]=difference(*base[:2],*curves[n][:2])
    fine=curves['M2_space'];d=np.load(P/'inputs/raw_DSMC_M2.npz',allow_pickle=False)
    dx,dc=centered(d['x'],d['rho'])
    old=np.genfromtxt(P/'inputs/fig1_M2.csv',delimiter=',',names=True)
    ox,oc=centered(old['x'],old['rho_from_f'])
    report={'model':'Original BGK, Pr=1, mu proportional T; Fei Fig4(a) model',
       'physical_reference':manifest['physical_reference'],
       'stationarity':{n:{k:s[k] for k in ('time','final_rhs','last_drift','stable','min_reduced','min_rho','min_T','face_flux_relative_spans')} for n,s in states.items()},
       'independent_original_numpy_step_final_RHS':original_checks,
       'density_sensitivity':grid,'density_sensitivity_0p5percent_jump_pass':all(g['max_percent_density_jump']<=.5 for g in grid.values()),
       'matched_BGK_vs_archived_DSMC':difference(*fine[:2],dx,d['rho']),
       'historical_constant_tau_BGK_vs_archived_DSMC':difference(ox,old['rho_from_f'],dx,d['rho']),
       'input_hashes':{n:sha(P/'inputs'/n) for n in ('raw_DSMC_M2.npz','fig1_M2.csv')},
       'interpretation':'One spatial refinement pair measures sensitivity, not certified order/error. Archived DSMC is finite-time, single-seed and a different scattering campaign from Fei VSS alpha=2.1403. Pairwise differences are descriptive, not model error.',
       'Fei_comparison':'Original Figure4(a) reproduced separately; no numerical digitization or false quantitative agreement. Our x is divided by independently computed lambda_Fei/relaxation_length; only a midpoint translation is applied.',
       'midpoint_shifts':{n:v[2] for n,v in curves.items()},'DSMC_shift':dc,'historical_BGK_shift':oc}
    (P/'COMPARISON.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    xx=np.linspace(-15,15,1201)
    table=np.column_stack([xx,np.interp(xx,*fine[:2]),np.interp(xx,dx,d['rho']),np.interp(xx,ox,old['rho_from_f'])])
    np.savetxt(P/'density_comparison.csv',table,delimiter=',',header='x_relaxation_length,BGK_mu_T,DSMC_archive,BGK_historical_tau1',comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':16,'axes.labelsize':18,'legend.fontsize':14,'xtick.labelsize':14,'ytick.labelsize':14,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    ratio=manifest['physical_reference']['lambda_Fei_in_our_length_units']
    ax[0].plot(dx/ratio,d['rho'],color='.55',lw=1,label='DSMC archive: raw, 2000 PPC')
    ax[0].plot(ox/ratio,old['rho_from_f'],'k:',lw=2,label=r'Historical BGK: $\tau=1$')
    ax[0].plot(fine[0]/ratio,fine[1],color='#0072B2',lw=2.4,label=r'Matched BGK: $\mu\propto T$, $Pr=1$')
    ax[0].set(xlim=(-10,10),ylim=(.8,2.4),xlabel=r'$(x-x_s)/\lambda_{\mathrm{Fei},0}$',ylabel=r'$\rho/\rho_1$',title='Mach 2: density comparison')
    ax[0].grid(alpha=.2);ax[0].legend(loc='lower right',framealpha=.95)
    ax[1].imshow(plt.imread(P/'Fei_Figure4a_M2.png'));ax[1].axis('off');ax[1].set_title('Fei et al. (2020), Fig. 4(a)\nPublished curves; separate VSS DSMC')
    fig.savefig(P/'BGK_M2_density_vs_DSMC_Fei.pdf',metadata={'Creator':'OpenAI Codex; data and citation provenance in COMPARISON.json'})
    fig.savefig(P/'BGK_M2_density_vs_DSMC_Fei.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    ratio=manifest['physical_reference']['lambda_Fei_in_our_length_units']
    for n,(x,rho,_) in curves.items():ax.plot(x/ratio,rho,lw=1.8,label=n)
    ax.set(xlim=(-10,10),ylim=(.9,2.4),xlabel=r'$(x-x_s)/\lambda_{\mathrm{Fei},0}$',ylabel=r'$\rho/\rho_1$',title='Stationary BGK: numerical sensitivity')
    ax.legend(loc='lower right');ax.grid(alpha=.2)
    fig.savefig(P/'BGK_M2_density_sensitivity.pdf',metadata={'Creator':'OpenAI Codex'})
    plt.close(fig)
    print(json.dumps(report['density_sensitivity']))
    return 0
if __name__=='__main__':sys.exit(main())
