"""Build a readable, source-tracked M2-M5 shock figure set from frozen profiles."""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfWriter

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
OUT=WORK/'output'/'pdf'/'shock_comparisons_20261010'
OUT.mkdir(parents=True,exist_ok=True)
HIGH=ROOT/'collected'; BASE=ROOT/'baseline_100ppc'
M2=WORK/'revision_runs'/'r26_M2_M3_comparison_20261009'/'aligned_profiles.npz'
M35=WORK/'revision_runs'/'dvm_dsmc_comparison_20261009'/'aligned_profiles.npz'
PAPER=WORK/'output'/'pdf'/'M2_R13_Yen1966_comparison_20261009'/'M2_R13_and_Yen1966.pdf'
PAPERDATA=WORK/'revision_runs'/'M2_R13_Yen1966_comparison_20261009'
PUBLISHED=WORK/'revision_runs'/'r26_M2_R13_Yen_M8_20261009'/'published_comparison.json'
DIGIT=WORK/'revision_runs'/'r26_M2_R13_Yen_M8_20261009'/'digitized_v4'
FIELDS=('rho','u','T','sigma','q','m','R','Delta','S')
M35FIELDS=('rho','T','q','R','Delta','S')
COL={'low':'#9AA0A6','raw':'#8BCDBB','smooth':'#007A5E','dvm':'#0072B2',
     'r26':'#D55E00','old':'#CC79A7','r26m3':'#009E73'}
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
    'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':11,
    'pdf.fonttype':42,'svg.fonttype':'none'})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_npz(p):
    with np.load(p,allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
        if 'fields' in d and 'columns' in d:
            d.update({str(k):d['fields'][:,i].copy() for i,k in enumerate(d['columns'])})
    if 'T' not in d and 'theta' in d:d['T']=d['theta']
    return d

def center(d,m):
    rho2=4*m*m/(m*m+3); target=(1+rho2)/2; y=d['rho']-target
    ids=np.flatnonzero(y[:-1]*y[1:]<=0); assert len(ids)
    roots=d['x'][ids]-y[ids]*np.diff(d['x'])[ids]/(y[ids+1]-y[ids])
    return float(roots[np.argmin(np.abs(roots))])

def local_quad(x,y,width=.75):
    dx=float(np.median(np.diff(x))); n=max(5,int(round(width/dx))); n+=1-(n%2)
    r=n//2; out=np.empty(len(y));
    for i in range(len(y)):
        lo=max(0,i-r);hi=min(len(y),i+r+1);s=max(abs(x[hi-1]-x[lo]),dx)
        t=(x[lo:hi]-x[i])/s; a=np.column_stack((np.ones(hi-lo),t,t*t))
        out[i]=np.linalg.lstsq(a,y[lo:hi],rcond=None)[0][0]
    return out,n

def save(fig,stem):
    for ext in ('png','pdf','svg'):fig.savefig(stem.with_suffix('.'+ext),dpi=240,bbox_inches='tight')
    plt.close(fig)

def m2_r13_yen_figure(high):
    # Replot the digitised R13 and Yen comparisons with the 2000-ppc M2 DSMC data.
    meta=json.loads(PUBLISHED.read_text(encoding='utf-8'))
    model_paths={
        'Corrected R26 Shakhov':WORK/'revision_runs'/'r26_M2_M3_comparison_20261009'/'M2_R26_Shakhov_p4q8'/'profile.npz',
        'Shakhov DVM, finite time':WORK/'revision_runs'/'dvm_r26_transport_match_20261007'/'completed_analysis'/'DVM_profiles'/'M2_x1280'/'profiles.npz',
        'Bird DSMC, 100 ppc':WORK/'revision_runs'/'dsmc_dvm_r26_M2_comparison_20261008'/'profiles.npz',
        'Previous R26 Maxwell (pre-STF)':WORK/'revision_runs'/'dvm_r26_transport_match_20261007'/'completed_analysis'/'R26_M2_N641_profile.npz'}
    center_keys={'Corrected R26 Shakhov':'R26 Shakhov, corrected',
        'Shakhov DVM, finite time':'DVM Shakhov, finite time',
        'Bird DSMC, 100 ppc':'Bird DSMC, VHS omega=1',
        'Previous R26 Maxwell (pre-STF)':'Old R26 Maxwell, pre-STF'}
    ds={}
    for name,p in model_paths.items():
        d=load_npz(p); d['x_lam']=(d['x']-meta['centers_our_units'][center_keys[name]])/meta['lambda_over_our_length'];
        if 'T' not in d:d['T']=d['theta']
        ds[name]=d
    h=dict(high); h['T']=h['theta']; h['x_lam']=(high['x']-center(high,2))/meta['lambda_over_our_length']; ds['Bird DSMC, 2000 ppc']=h
    # Printed R13 curves are digitised; use the source's saved shock-centre registration.
    papers={}
    for label,prefix in [('R13 linear, Timokhin 2015','2015_linear'),('R13 nonlinear, Timokhin 2015','2015_nonlinear'),('R13 full, Timokhin 2017','2017_2003full')]:
        xr=np.genfromtxt(DIGIT/(prefix+'_rho.csv'),delimiter=',',names=True)
        xt=np.genfromtxt(DIGIT/(prefix+'_T.csv'),delimiter=',',names=True)
        cc=meta['centers_papers_x_over_lambda'][label.replace('Timokhin 2015','Timokhin2015').replace('Timokhin 2017','Timokhin2017').replace('R13 full,','R13 2003full,')]
        papers[label]={'xr':xr['x_over_lambda']-cc,'rho':xr['value'],'xt':xt['x_over_lambda']-cc,'T':xt['value']}
    M=2.; rho2=16/7; T2=133/64; eta=np.linspace(1,rho2,600)
    Tx=((23/eta)-(20/eta**2))/3
    Tms=(35+23/eta-40/eta**2)/18
    cols={'Corrected R26 Shakhov':'#0072B2','Shakhov DVM, finite time':'#222222','Bird DSMC, 100 ppc':'#999999','Bird DSMC, 2000 ppc':'#007A5E','Previous R26 Maxwell (pre-STF)':'#CC79A7'}
    fig,axs=plt.subplots(2,2,figsize=(15,10)); axs=axs.ravel()
    # Density and mean temperature against registered position.
    for ax,key,title in [(axs[0],'rho','Density profile'),(axs[1],'T','Mean temperature profile')]:
        jump=rho2-1 if key=='rho' else T2-1
        for name,d in ds.items():
            xx=d['x_lam']; yy=(d[key]-1)/jump; mask=np.abs(xx)<=13
            if name=='Bird DSMC, 2000 ppc':
                ax.scatter(xx[mask][::12],yy[mask][::12],s=8,color='#8BCDBB',alpha=.48,label='DSMC 2000 ppc raw')
                sm,w=local_quad(xx[mask],yy[mask],.75); ax.plot(xx[mask],sm,color=cols[name],lw=1.8,label='DSMC 2000 ppc smooth (display only)')
            else: ax.plot(xx[mask],yy[mask],color=cols[name],lw=1.4,label=name)
        for name,d in papers.items():
            xx=d['xr'] if key=='rho' else d['xt']; yy=d[key]
            ax.plot(xx,yy,lw=1.1,marker={'R13 linear, Timokhin 2015':'o','R13 nonlinear, Timokhin 2015':'^','R13 full, Timokhin 2017':'s'}[name],ms=3,markevery=max(1,len(xx)//24),label=name+' [digitised]')
        ax.set(xlim=(-12,12),ylim=(-.03,1.06),xlabel=r'$(x-x_{1/2})/\lambda_1$',ylabel='normalized jump',title=title); ax.grid(alpha=.22)
    # Exact conservation relation and Mott-Smith mean-temperature approximation.
    axs[2].plot(eta,Tx,color='black',lw=2,label='Yen 1966 Eq. 1: conservation relation')
    axs[3].plot(eta,Tms,color='black',lw=2,ls=':',label='Yen 1966 Eq. 5: Mott–Smith approximation')
    for name,d in ds.items():
        rho=d['rho']; T=d['T']; sig=d['sigma']; mask=(rho>=1.001)&(rho<=rho2-.001)
        ids=np.flatnonzero(mask)[::max(1,int(mask.sum()/45))]
        if name=='Bird DSMC, 2000 ppc':
            axs[2].scatter(rho[ids],(T+sig/rho)[ids],s=9,color='#8BCDBB',alpha=.55,label='DSMC 2000 ppc raw')
            axs[3].scatter(rho[ids],T[ids],s=9,color='#8BCDBB',alpha=.55,label='_nolegend_')
        else:
            axs[2].plot(rho[ids],(T+sig/rho)[ids],'.',ms=3,color=cols[name],label=name)
            axs[3].plot(rho[ids],T[ids],color=cols[name],lw=1.3,label=name)
    for name,d in papers.items():
        mask=(d['xt']>=d['xr'].min())&(d['xt']<=d['xr'].max())
        rho_at_t=1+(rho2-1)*np.interp(d['xt'][mask],d['xr'],d['rho'])
        temp_at_t=1+(T2-1)*d['T'][mask]
        axs[3].plot(rho_at_t,temp_at_t,lw=1.1,marker={'R13 linear, Timokhin 2015':'o','R13 nonlinear, Timokhin 2015':'^','R13 full, Timokhin 2017':'s'}[name],ms=2.5,markevery=max(1,len(temp_at_t)//24),label=name+' [digitised]')
    axs[2].set(xlim=(1,rho2),xlabel=r'$\rho/\rho_1$',ylabel=r'$T_{xx}/T_1$',title='Yen: longitudinal temperature');axs[3].set(xlim=(1,rho2),xlabel=r'$\rho/\rho_1$',ylabel=r'$T/T_1$',title='Mean temperature at equal density')
    for ax in axs[2:]:ax.grid(alpha=.22)
    handles,labels=axs[0].get_legend_handles_labels();
    # Include entries whose plotted data appear only in the lower panels.
    h2,l2=axs[2].get_legend_handles_labels();
    for hh,ll in zip(h2,l2):
        if ll not in labels:handles.append(hh);labels.append(ll)
    h3,l3=axs[3].get_legend_handles_labels()
    for hh,ll in zip(h3,l3):
        if ll not in labels:handles.append(hh);labels.append(ll)
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015),fontsize=10,frameon=True)
    fig.suptitle('Mach 2 | corrected R26, DSMC, DVM, published R13 and Yen (1966)',fontsize=15)
    fig.text(.5,.125,'R13 curves digitised from published plots. Yen Eq. 1 is a conservation identity; Eq. 5 is a Mott–Smith approximation, not independent Boltzmann data. Translation only; no width fit.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.22,1,.95))
    save(fig,OUT/'04_M2_R13_Yen_all_answers')

def main():
    old=load_npz(M2); m35=load_npz(M35); x=old['x']; assert np.array_equal(x,m35['x'])
    aligned={}; windows={}; hashes={}
    for m in range(2,6):
        d=load_npz(HIGH/f'M{m}'/'profiles.npz'); hashes[f'M{m}_DSMC_2000ppc']=sha(HIGH/f'M{m}'/'profiles.npz')
        c=center(d,m); xc=d['x']-c; item={'x':xc}
        for k in FIELDS:
            src='theta' if k=='T' else k; item[k]=d[src]
            item[k+'_smooth'],w=local_quad(xc,item[k]);windows[str(m)]=w
            item[k]=np.interp(x,xc,item[k]); item[k+'_smooth']=np.interp(x,xc,item[k+'_smooth'])
        aligned[m]=item

    m2_r13_yen_figure(load_npz(HIGH/'M2'/'profiles.npz'))

    # M2: all available nine-field profiles, with both DSMC particle counts.
    fig,axs=plt.subplots(3,3,figsize=(15,11),sharex=True,constrained_layout=True)
    for ax,k in zip(axs.flat,FIELDS):
        ax.plot(x,old[f'M2_DSMC_{k}'],color=COL['low'],lw=.8,alpha=.8,label='Bird DSMC, 100 ppc raw')
        ax.scatter(x[::5],aligned[2][k][::5],s=7,color=COL['raw'],alpha=.55,label='Bird DSMC, 2000 ppc raw')
        ax.plot(x,aligned[2][k+'_smooth'],color=COL['smooth'],lw=1.8,label='Bird DSMC, 2000 ppc smoothed')
        ax.plot(x,old[f'M2_DVM_{k}'],color=COL['dvm'],ls='--',lw=1.5,label='Shakhov DVM, finite-time snapshot')
        ax.plot(x,old[f'M2_Previous R26 Maxwell_{k}'],color=COL['old'],ls=':',lw=1.6,label='Previous R26 Maxwell, pre-STF repair')
        ax.plot(x,old[f'M2_New R26 Shakhov_{k}'],color=COL['r26'],lw=1.7,label='Corrected R26 Shakhov')
        ax.set(xlim=(-15,15),xlabel=r'$x-x_{1/2}$ ($\lambda_1$)',ylabel=k);ax.grid(alpha=.22)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.035),frameon=True,fontsize=10.5)
    fig.suptitle('Mach 2 | all available nine-field comparisons',fontsize=16)
    fig.text(.5,-.083,'R26 previous Maxwell curve is pre-STF evidence; DVM is not independently stationary. Smoothing is display-only; all raw DSMC samples are retained.',ha='center',fontsize=9.5)
    save(fig,OUT/'01_M2_all_models_nine_fields')

    # M3-M5: high-count DSMC and available DVM; corrected R26 exists at M3 only.
    fig,axs=plt.subplots(3,6,figsize=(19,10.5),sharex=True,constrained_layout=True)
    for r,m in enumerate((3,4,5)):
        src=m35
        for c,k in enumerate(M35FIELDS):
            ax=axs[r,c]
            ax.plot(x,aligned[m][k],color='#AEB4B8',lw=.65,alpha=.65,label='DSMC 2000 ppc raw' if r==c==0 else None)
            ax.scatter(x[::5],aligned[m][k][::5],s=5,color=COL['raw'],alpha=.45,label='_nolegend_')
            ax.plot(x,aligned[m][k+'_smooth'],color=COL['smooth'],lw=1.7,label='DSMC 2000 ppc smooth' if r==c==0 else None)
            ax.plot(x,src[f'M{m}_DVM_{k}'],color=COL['dvm'],lw=1.55,ls='--',label='Shakhov DVM snapshot' if r==c==0 else None)
            rk=f'M{m}_corrected_Shakhov_R26_{k}'
            if rk in src:
                ax.plot(x,src[rk],color=COL['r26m3'],lw=1.6,label='Corrected Shakhov R26' if r==c==0 else None)
            ax.grid(alpha=.2)
            if r==0:ax.set_title(k)
            if c==0:ax.set_ylabel(f'Mach {m}\n{k}')
            if r==2:ax.set_xlabel(r'$x-x_{1/2}$ ($\lambda_1$)')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.025),fontsize=11,frameon=True)
    fig.suptitle('Mach 3–5 | high-particle DSMC, Shakhov DVM, and accepted R26 where available',fontsize=16)
    fig.text(.5,-.067,'No accepted corrected R26 profile is available at Mach 4 or 5. DVM profiles are transient snapshots; DSMC stationarity is uncertified. Smoothing is display-only.',ha='center',fontsize=9.5)
    save(fig,OUT/'02_M3_M5_DSMC_DVM_R26')

    # Channel-error audit: signed component errors and their sum, common physical scale.
    audit={}
    fig,axs=plt.subplots(2,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for row,m in enumerate((2,3)):
        ref=aligned[m]; r26={k:old[f'M{m}_New R26 Shakhov_{k}'] for k in ('R','Delta','S')}
        scale=float(np.sqrt(np.mean(ref['S']**2)))
        er=(r26['R']-ref['R'])/scale; ed=((r26['Delta']-ref['Delta'])/3)/scale
        es=(r26['S']-ref['S'])/scale
        assert np.max(np.abs(es-er-ed))<1e-10
        nr=np.linalg.norm(er);nd=np.linalg.norm(ed);ns=np.linalg.norm(es)
        corr=float(np.dot(er,ed)/(nr*nd))
        vals={'R_component_L2_percent':100*nr/np.sqrt(len(x)),
              'Delta_over_3_component_L2_percent':100*nd/np.sqrt(len(x)),
              'combined_S_L2_percent':100*ns/np.sqrt(len(x)),
              'signed_error_correlation':corr,
              'identity_max_abs':float(np.max(np.abs(es-er-ed))),
              'common_scale':'RMS of DSMC S on the common x grid'}
        audit[str(m)]=vals
        ax=axs[row]
        ax.plot(x,er,color='#0072B2',lw=1.6,label=r'$e_R$' if row==0 else None)
        ax.plot(x,ed,color='#D55E00',lw=1.6,label=r'$e_{\Delta}/3$' if row==0 else None)
        ax.plot(x,es,color='#009E73',lw=1.9,label=r'$e_S=e_R+e_{\Delta}/3$' if row==0 else None)
        ax.axhline(0,color='#555',lw=.7);ax.grid(alpha=.22)
        ax.set_ylabel(f'Mach {m}\nerror / RMS($S_{{DSMC}}$)')
        ax.text(.99,.94,f'R: {vals["R_component_L2_percent"]:.1f}%   Δ/3: {vals["Delta_over_3_component_L2_percent"]:.1f}%   S: {vals["combined_S_L2_percent"]:.1f}%\nerror correlation: {corr:+.2f}',transform=ax.transAxes,ha='right',va='top',fontsize=10,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    axs[0].legend(loc='upper left',ncol=3,fontsize=11,frameon=True)
    axs[-1].set_xlabel(r'$x-x_{1/2}$ ($\lambda_1$)')
    fig.suptitle('Closure-channel error decomposition | corrected Shakhov R26 minus high-particle DSMC',fontsize=14)
    fig.text(.5,-.025,'Same reference scale for all three signed errors. This is an error-geometry diagnostic, not a causal shoulder test; DSMC sampling and model/BC differences remain.',ha='center',fontsize=9)
    save(fig,OUT/'03_closure_channel_signed_errors_M2_M3')
    (OUT/'closure_channel_error_audit.json').write_text(json.dumps({'definition':'e_S=e_R+e_Delta/3; all components use the same RMS(S_DSMC) scale; density-midpoint translation only','cases':audit,'interpretation_limit':'The observation-map null direction can hide component errors in a heat-flux-budget comparison, but full R26 carries separate R and Delta equations. This profile comparison cannot explain nonlinear solver convergence or prove the channel causes shock shoulders. Different collision operators, finite DSMC sampling, and BCs remain.'},indent=2),encoding='utf-8')

    # Include source-verified individual panels plus focused convergence/count figures.
    supporting=[
      PAPER,
      WORK/'revision_runs'/'r26_M2_M3_comparison_20261009'/'M3_nine_fields.pdf',
      WORK/'revision_runs'/'dsmc_dvm_r26_M2_comparison_20261008'/'M2_three_methods.pdf',
      WORK/'output'/'pdf'/'shock_comparisons_20261007'/'02_M2_DVM_three_grids.pdf',
      WORK/'output'/'pdf'/'shock_comparisons_20261007'/'03_M2_DVM_parameter_sensitivity.pdf',
      WORK/'output'/'pdf'/'shock_comparisons_20261007'/'06_M2_R26_three_grids.pdf',
      WORK/'output'/'pdf'/'shock_comparisons_20261007'/'07_DSMC_particle_count_M2_M5.pdf',
      OUT/'01_M2_all_models_nine_fields.pdf',OUT/'02_M3_M5_DSMC_DVM_R26.pdf',
      OUT/'03_closure_channel_signed_errors_M2_M3.pdf',OUT/'04_M2_R13_Yen_all_answers.pdf']
    # Create a multipage PDF whose first pages are the highest-value all-model figures.
    writer=PdfWriter()
    for p in supporting:
        if p.exists():writer.append(str(p))
    with (OUT/'Shock_figures_M2_to_M5.pdf').open('wb') as f:writer.write(f)
    writer.close()
    # ZIP the readable figures and metadata/report; no scientific source data are altered.
    selected=[]
    for stem in ('01_M2_all_models_nine_fields','02_M3_M5_DSMC_DVM_R26',
                 '03_closure_channel_signed_errors_M2_M3','04_M2_R13_Yen_all_answers'):
        selected += [OUT/f'{stem}.{e}' for e in ('png','pdf','svg')]
    selected += [p for p in supporting if p.exists() and p.suffix=='.pdf']
    shutil.copy2(OUT/'Shock_figures_M2_to_M5.pdf',OUT/'Shock_figures_M2_to_M5_bundle.pdf')
    # Keep one canonical multipage PDF in the archive rather than duplicating it.
    selected=[p for p in selected if p.name!='Shock_figures_M2_to_M5_bundle.pdf']+[OUT/'Shock_figures_M2_to_M5.pdf']
    readme=(OUT/'README_FA.txt')
    readme.write_text('''مجموعهٔ شکل‌های شوک ماخ ۲ تا ۵\n\n01: ماخ۲، ۹ میدان؛ DSMC با ۱۰۰ و ۲۰۰۰ ذره/سلول، DVM، R26 Shakhov اصلاح‌شده، و R26 Maxwell پیش از اصلاح STF.\n02: ماخ۳ تا ۵؛ DSMC پرذره و DVM؛ R26 اصلاح‌شده فقط برای ماخ۳ موجود است.\n03: تجزیهٔ خطای امضادار کانال $S=R+\\Delta/3$ برای R26 نسبت به DSMC در ماخ۲ و۳، با مقیاس مشترک.\n04: ماخ۲ با سه منحنی منتشرشدهٔ R13 و روابط Yen؛ منحنی‌های R13 دیجیتایز شده‌اند.\nشکل‌های همگرایی شبکه و حساسیت DVM ماخ۲ و مقایسهٔ تعداد ذرات هم در همان PDF آمده‌اند. راهنماهای شکل‌های جدید بزرگ‌تر شده‌اند. دادهٔ خام DSMC حفظ شده؛ هموارسازی موضعی درجه‌دو فقط برای نمایش است. منحنی «R26 Maxwell پیشین» مدرک تاریخی پیش از اصلاح STF است و نباید به‌عنوان اجرای معتبر فعلی خوانده شود. DVMهای ماخ۳ تا۵ snapshot گذرا هستند و ایستایی DSMC تأیید نشده است.\nتراز مکانی فقط انتقال بر اساس میانهٔ چگالی Rankine–Hugoniot است؛ پهنا برازش نشده. شکل‌ها به‌تنهایی اعتبار فیزیکی یا همگرایی آماری را ثابت نمی‌کنند.\n''',encoding='utf-8')
    provenance={'created_from':{str(p.relative_to(WORK)):sha(p) for p in [M2,M35]+[HIGH/f'M{m}'/'profiles.npz' for m in range(2,6)]},
        'high_ppc_smoothing':{'method':'centered local quadratic least squares','nominal_width_lambda1':.75,'window_points':windows},
        'legend_font_points':{'M2':10.5,'M3_M5':11},'supporting_pdfs':[str(p) for p in supporting if p.exists()],
        'outputs':{p.name:sha(p) for p in selected if p.exists()},
        'limits':['M4 and M5 accepted R26 profiles unavailable.','DSMC stationarity is uncertified.','DVM M3-M5 is transient.','Display smoothing does not change raw data or quantitative metrics.']}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    import zipfile
    zpath=OUT/'Shock_figures_M2_to_M5.zip'
    members=set(selected+[OUT/'README_FA.txt',OUT/'provenance.json',OUT/'closure_channel_error_audit.json'])
    members.update(OUT/f'03_closure_channel_signed_errors_M2_M3.{e}' for e in ('png','pdf','svg'))
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for p in sorted(members):
            if p.exists():zf.write(p,p.name)
    print(json.dumps({'folder':str(OUT),'multipage_pdf':str(OUT/'Shock_figures_M2_to_M5.pdf'),
       'zip':str(zpath),'figure_count':len(selected),'supporting_pages':len(supporting),'windows':windows,'channel_audit':audit,'outputs':provenance['outputs']},indent=2))

if __name__=='__main__':main()
