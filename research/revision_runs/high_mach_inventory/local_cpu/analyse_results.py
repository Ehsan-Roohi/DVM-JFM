"""Independent finite/hash checks and historical moment diagnostic figures."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
tags=['M6','M8','M12','M8lite','M12lite']
data={};audits={}
for tag in tags:
    p=ROOT/tag
    done=json.loads((p/'DONE.json').read_text())
    for name,h in done['files'].items():
        assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h
    z=np.load(p/'moments.npz');data[tag]={k:z[k] for k in z.files}
    for k in ['rho','ux','T','qx','sig','Rxx_closure','Delta_dvm','S_dvm','Delta_raw']:
        assert np.isfinite(z[k]).all(),(tag,k)
    a=done['audit'];a['plateau_diagnostics']={}
    for k in ['Rxx_closure','Delta_dvm','S_dvm']:
        y=z[k];n=max(5,len(y)//20);scale=max(float(np.max(np.abs(y))),1e-30)
        a['plateau_diagnostics'][k]={'left_mean':float(y[:n].mean()),'right_mean':float(y[-n:].mean()),'left_relative_to_peak':float(y[:n].mean()/scale),'right_relative_to_peak':float(y[-n:].mean()/scale)}
    audits[tag]=a
comparisons={}
for tag in ['M8','M12']:
    a=data[tag];b=data[tag+'lite'];x=a['x_mfp'];mask=(x>=b['x_mfp'][0])&(x<=b['x_mfp'][-1]);r={}
    for k in ['Rxx_closure','Delta_dvm','S_dvm']:
        u=a[k][mask];v=np.interp(x[mask],b['x_mfp'],b[k])
        r[k]=float(np.linalg.norm(u-v)/max(np.linalg.norm(u),1e-30))
    comparisons[tag]=r
out={'audits':audits,'full_lite_relative_l2_on_stored_x_mfp':comparisons,'interpretation':'Historical BGK data. Discrete-reference Delta/S and raw continuum Delta remain distinct. No plateau correction applied. Full/lite changes multiple parameters, so differences are sensitivity diagnostics, not isolated convergence. Source association is based on preserved submitters/current hashes, not immutable execution-time source manifest.'}
(ROOT/'analysis.json').write_text(json.dumps(out,indent=2))
fig,axs=plt.subplots(3,1,figsize=(8,10))
for ax,key,label in zip(axs,['Rxx_closure','Delta_dvm','S_dvm'],['R','Delta (discrete reference)','S = R + Delta/3']):
    for tag in tags:
        d=data[tag];ax.plot(d['x_mfp'],d[key],label=tag,ls='--' if 'lite' in tag else '-')
    ax.set_xlabel('Stored x / upstream mean free path');ax.set_ylabel(label);ax.grid(alpha=.2)
axs[0].legend(ncol=3,fontsize=11)
fig.suptitle('Historical BGK moments: uncorrected plateau diagnostics',fontsize=13)
fig.tight_layout();fig.savefig(ROOT/'historical_moments.pdf');fig.savefig(ROOT/'historical_moments.png',dpi=180)
lines=['تحلیل محلی داده‌های تاریخی BGK؛ بدون تصحیح سکوی دوردست.','هش خروجی‌ها و متناهی‌بودن میدان‌ها مستقلاً بررسی شد. اختلاف full/lite آزمون حساسیت است، نه اثبات همگرایی تک‌پارامتری.']
for tag,a in audits.items():
    lines.append(f"{tag}: minimum f={a['minimum_f']:.6g}; Delta iteration relative L2={a['Delta_dvm_iteration_relative_l2']:.6g}; stored R relative L2={a['Rxx_closure_stored_relative_l2']:.6g}")
lines.append('شکل‌ها تشخیصی‌اند؛ پیش از درج در مقاله باید خطای تطبیق، اثر سکو و حساسیت پارامترها بررسی شود.')
(ROOT/'analysis_FA.txt').write_text('\n'.join(lines),encoding='utf-8')
