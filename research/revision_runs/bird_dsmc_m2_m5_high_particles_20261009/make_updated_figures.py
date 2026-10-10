"""Update DSMC shock plots with 100/2000-ppc results and display-only smoothing."""
from pathlib import Path
import csv, hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
HIGH = ROOT / 'collected'
BASE = ROOT / 'baseline_100ppc'
OUT = ROOT.parent / 'dsmc_dvm_r26_M2_comparison_20261008'
PLOT = ROOT.parent.parent / 'output' / 'pdf' / 'shock_comparisons_20261007'
OUT.mkdir(parents=True, exist_ok=True)
PLOT.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
FIELDS = ['rho','u','T','sigma','q','R','Delta','S']
LABELS = [r'$\rho$',r'$u_x$',r'$T$',r'$\sigma_{xx}$',r'$q_x$',r'$R_{xx}^{cl}$',r'$\Delta$',r'$S$']
COLORS = {'low':'#92999F','high_raw':'#91CDBB','high_smooth':'#007A5E',
          'dvm':'#0072B2','r26':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                      'font.size':10,'axes.labelsize':10,'axes.titlesize':11,
                      'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                      'axes.spines.right':False})

def read(path):
    z=np.load(path,allow_pickle=False)
    d={k:z[k].copy() for k in z.files}
    if 'theta' in d: d['T']=d['theta']
    assert all(np.isfinite(d[k]).all() for k in ['x']+FIELDS),path
    assert np.all(np.diff(d['x'])>0) and np.min(d['rho'])>0 and np.min(d['T'])>0
    return d

def midpoint_x(d,mach):
    # Rankine-Hugoniot density midpoint; translate only, with linear crossing interpolation.
    ratio=4.0*mach**2/(mach**2+3.0)
    target=(1.0+ratio)/2.0
    y=d['rho']-target
    ii=np.flatnonzero(y[:-1]*y[1:]<=0)
    assert len(ii)>0,(mach,'no density-midpoint crossing')
    i=ii[np.argmin(np.abs(d['x'][ii]))]
    xs=d['x'][i]-y[i]*(d['x'][i+1]-d['x'][i])/(y[i+1]-y[i])
    return xs

def smooth_local_quadratic(x,y,width=.75):
    """Centered local quadratic least squares over ~0.75 upstream mean-free-path."""
    dx=float(np.median(np.diff(x)))
    nwin=max(5,int(round(width/dx)))
    if nwin%2==0:nwin+=1
    n=len(y);r=nwin//2;out=np.empty_like(y,dtype=float)
    for i in range(n):
        lo=max(0,i-r);hi=min(n,i+r+1)
        s=max(abs(x[hi-1]-x[lo]),dx)
        t=(x[lo:hi]-x[i])/s
        A=np.column_stack((np.ones(hi-lo),t,t*t))
        out[i]=np.linalg.lstsq(A,y[lo:hi],rcond=None)[0][0]
    return out,nwin

def plot_style(ax):
    ax.grid(color='#DDE3E8',lw=.6,alpha=.85);ax.set_axisbelow(True)
    ax.axhline(0,color='#AAB2B9',lw=.55,zorder=0)
    ax.tick_params(direction='out',length=2.5)

def read_mach_set(mach):
    low=read(BASE/f'M{mach}'/'profiles.npz')
    high=read(HIGH/f'M{mach}'/'profiles.npz')
    for d in (low,high):
        xs=midpoint_x(d,mach);d['xc']=d['x']-xs
    high['smooth']={};windows={}
    for k in FIELDS:
        high['smooth'][k],windows[k]=smooth_local_quadratic(high['x'],high[k])
    assert len(set(windows.values()))==1
    return low,high,windows['rho']

def read_m2_models():
    dvm_path=ROOT.parent/'dvm_r26_transport_match_20261007'/'completed_analysis'/'DVM_profiles'/'M2_x1280'/'profiles.npz'
    r26_path=ROOT.parent/'dvm_r26_transport_match_20261007'/'completed_analysis'/'R26_M2_N641_profile.npz'
    z=np.load(dvm_path,allow_pickle=False);dvm={k:z[k].copy() for k in z.files}
    dvm.update({str(k):z['fields'][:,i].copy() for i,k in enumerate(z['columns'])})
    if 'theta' in dvm:dvm['T']=dvm['theta']
    xs=midpoint_x(dvm,2);dvm['xc']=dvm['x']-xs
    z=np.load(r26_path,allow_pickle=False);r26={k:z[k].copy() for k in z.files}
    r26['T']=r26['theta'];xs=midpoint_x(r26,2);r26['xc']=r26['x']-xs
    for d in (dvm,r26):
        assert all(np.isfinite(d[k]).all() for k in FIELDS)
    return dvm,r26,dvm_path,r26_path

sets={m:read_mach_set(m) for m in range(2,6)}
for m,(low,high,w) in sets.items():
    print(f'M{m}: local quadratic display window {w} points (~0.75 lambda_1)')

# M2: update the established three-method figure while retaining its old version.
oldstem=OUT/'M2_three_methods'
for ext in ('png','pdf','svg'):
    old=oldstem.with_suffix('.'+ext)
    if old.exists():
        backup=OUT/f'M2_three_methods_100ppc_original.{ext}'
        if not backup.exists():shutil.copy2(old,backup)
low,high,win=sets[2];dvm,r26,dvm_path,r26_path=read_m2_models()
fig,axs=plt.subplots(2,4,figsize=(15,7.2),constrained_layout=True)
for ax,k in zip(axs.flat,FIELDS):
    ax.plot(low['xc'],low[k],color=COLORS['low'],lw=.75,alpha=.65,label='Bird DSMC: 100 ppc, raw')
    ax.scatter(high['xc'][::3],high[k][::3],s=5,color=COLORS['high_raw'],alpha=.48,
               linewidths=0,label='Bird DSMC: 2000 ppc, raw')
    ax.plot(high['xc'],high['smooth'][k],color=COLORS['high_smooth'],lw=1.65,
            label='Bird DSMC: 2000 ppc, smoothed')
    ax.plot(dvm['xc'],dvm[k],color=COLORS['dvm'],lw=1.55,label='Shakhov DVM: 1280 cells')
    ax.plot(r26['xc'],r26[k],color=COLORS['r26'],lw=1.55,ls='--',label='R26: 641 nodes')
    ax.set(xlim=(-15,15),xlabel=r'$x-x_s$ ($\lambda_1$)',ylabel=k)
    plot_style(ax)
axs.flat[0].legend(fontsize=7.2,loc='best',frameon=True)
fig.suptitle('Mach 2 | R26, transport-matched Shakhov DVM and Bird DSMC',fontsize=14)
fig.text(.5,-.018,'DSMC smoothing is display-only (local quadratic, 0.75 $\lambda_1$ window). Quantitative differences use raw profiles; density-midpoint translation only, no width fit. DSMC stationarity is not certified.',ha='center',fontsize=8.3,color='#505B64')
fig.savefig(oldstem.with_suffix('.png'),dpi=210,bbox_inches='tight')
fig.savefig(oldstem.with_suffix('.pdf'),bbox_inches='tight')
fig.savefig(oldstem.with_suffix('.svg'),bbox_inches='tight')
plt.close(fig)

# Four-Mach DSMC particle-count plot: show low-count raw, high-count raw, and smoothed high-count.
fields4=['rho','T','q','Delta'];labels4=[r'$\rho$',r'$T$',r'$q_x$',r'$\Delta$']
fig,axs=plt.subplots(4,4,figsize=(15,12),sharex=True,constrained_layout=True)
for row,m in enumerate(range(2,6)):
    low,high,w=sets[m]
    for col,(k,label) in enumerate(zip(fields4,labels4)):
        ax=axs[row,col]
        ax.plot(low['xc'],low[k],color=COLORS['low'],lw=.7,alpha=.62,label='100 ppc raw')
        ax.scatter(high['xc'][::3],high[k][::3],s=4,color=COLORS['high_raw'],alpha=.48,
                   linewidths=0,label='2000 ppc raw')
        ax.plot(high['xc'],high['smooth'][k],color=COLORS['high_smooth'],lw=1.5,
                label='2000 ppc smoothed')
        ax.set_xlim(-15,15);plot_style(ax)
        if row==0:ax.set_title(label)
        if col==0:ax.set_ylabel(f'M={m}\n{label}')
        if row==3:ax.set_xlabel(r'$x-x_s$ ($\lambda_1$)')
axs[0,0].legend(fontsize=7,loc='best')
fig.suptitle('Bird DSMC | 100 versus 2000 upstream particles per cell',fontsize=15)
fig.text(.5,-.012,'Same 800-cell grid, dt=0.005 and t=400; independent seeds. Density-midpoint translation only. Local quadratic 0.75 $\lambda_1$ smoothing is for display; raw profiles and finite-sample bias are retained.',ha='center',fontsize=8.5,color='#505B64')
newstem=PLOT/'07_DSMC_particle_count_M2_M5'
for ext in ('png','pdf','svg'):
    fig.savefig(newstem.with_suffix('.'+ext),dpi=210,bbox_inches='tight')
plt.close(fig)

# Raw-profile comparison metrics (no smoothing in reported numbers).
grid=np.linspace(-15,15,1201);metrics={};count_sensitivity={}
for name,d in [('DVM',dvm),('R26',r26)]:
    metrics[name]={}
    for k in FIELDS:
        a=np.interp(grid,sets[2][1]['xc'],sets[2][1][k])
        b=np.interp(grid,d['xc'],d[k])
        metrics[name][k]=float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
count_sensitivity={}
for k in FIELDS:
    low=np.interp(grid,sets[2][0]['xc'],sets[2][0][k])
    high=np.interp(grid,sets[2][1]['xc'],sets[2][1][k])
    count_sensitivity[k]=float(100*np.linalg.norm(high-low)/np.linalg.norm(high))
inputs={}
for m in range(2,6):
    for group in ('baseline_100ppc','collected'):
        d=ROOT/group/f'M{m}'
        for n in ('profiles.npz','RECEIPT.json','status.json','analysis.json'):
            inputs[str((d/n).relative_to(ROOT))]=sha(d/n)
inputs[str(dvm_path.relative_to(ROOT.parent.parent))]=sha(dvm_path)
inputs[str(r26_path.relative_to(ROOT.parent.parent))]=sha(r26_path)
report={'cases':{str(m):{'blocks':800,'time':400,'upstream_ppc':2000,'baseline_ppc':100,
                         'stationarity_certified':False,'raw_profile_sha256':sha(HIGH/f'M{m}'/'profiles.npz'),
                         'display_smoothing':{'method':'centered local quadratic least squares',
                                              'nominal_width_lambda1':0.75,
                                              'window_points':sets[m][2]}}
                 for m in range(2,6)},
        'M2_raw_L2_difference_percent':metrics,
        'M2_raw_100_vs_2000_ppc_L2_percent':count_sensitivity,
        'alignment':'Rankine-Hugoniot density midpoint translation only; no width fit',
        'smoothing':'Visualization only; all raw data retained and raw profiles used for reported L2 metrics',
        'limitations':['Single independent high-particle seed per Mach; no autocorrelation-corrected confidence intervals.',
                       'DSMC finite-sample bias in fourth and higher central moments remains.',
                       'Different collision operators and boundary treatments mean model differences are not pure discretization error.',
                       'A smooth display curve is not a corrected DSMC estimator.'],
        'input_sha256':inputs}
(ROOT/'updated_figures_report.json').write_text(json.dumps(report,indent=2))
with (ROOT/'updated_figures_report_FA.txt').open('w',encoding='utf-8') as f:
    f.write('شکل‌های DSMC ماخ ۲ تا ۵ با داده‌های ۲۰۰۰ ذره در هر سلول به‌روز شدند.\n')
    f.write('هر نمودار منحنی خام را نگه می‌دارد و هموارسازی موضعی درجه‌دو با پنجره اسمی ۰٫۷۵ طول آزاد میانگین بالادست را فقط برای نمایش روی آن می‌گذارد.\n')
    f.write('مرکز شوک فقط با انتقال نقطهٔ میانی چگالی RH هم‌تراز شده؛ پهنا برازش نشده است. اعداد اختلاف مدل در ماخ ۲ از پروفایل خام به‌دست آمده‌اند.\n')
    f.write('هر چهار اجرا به t=400 و ۸۰۰ بلوک رسیده‌اند، اما ایستایی آماری تأیید نشده است. دادهٔ هموارشده برآورد تصحیح‌شدهٔ DSMC یا فاصلهٔ اطمینان نیست.\n\n')
    f.write('اختلاف خام ماخ ۲ بین ۱۰۰ و ۲۰۰۰ ذره در هر سلول (L2 نسبی، درصد):\n')
    f.write(json.dumps(count_sensitivity,indent=2,ensure_ascii=False)+'\n\n')
    f.write(json.dumps(metrics,indent=2))
print(json.dumps({'M2_raw_L2_difference_percent':metrics,'four_mach_figure':str(newstem),'updated_M2_figure':str(oldstem)},indent=2))
