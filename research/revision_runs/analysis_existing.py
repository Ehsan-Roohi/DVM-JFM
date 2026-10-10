"""Prespecified sparse-position stress tests and ESFP admissibility diagnostics.
No new collision physics and no full-model ESFP claim.
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'source'))
import sparse_delta_rbf_second_channel as rbf

ROOT=Path(__file__).resolve().parent
ORIG=ROOT.parent
OUT=ROOT/'outputs/existing_data_audit'
OUT.mkdir(parents=True,exist_ok=True)

def main():
    if (OUT/'DONE.json').exists(): return
    records=[]; sensors={}; admissibility=[]
    for mach in [2,3,5]:
        p=ORIG/f'appendix_validation_dvm/exact_heatflux_budget/M{mach}_exact_budget_xmfp_exact_heatflux_budget_profiles.csv'
        data=np.genfromtxt(p,delimiter=',',names=True)
        # Independent reconstruction must exist. No fallback to reference S.
        x=data['x']; R=rbf.plateau_zero(data['Rxx_closure'])
        D=rbf.plateau_zero(data['Delta']); S=rbf.plateau_zero(data['S_R_plus_Delta3_rec'])
        mask=rbf.active_mask(S,.02)
        active=rbf.active_mask(D,.02); lo,hi=x[active][[0,-1]]
        centre=(lo+hi)/2; width=hi-lo
        base=rbf.select_sensors(x,D,24,mode='active')
        for layout in ['baseline','jitter','gap10','gap25','gap40','whole_domain']:
            for noise in [0.,.005,.01]:
                for seed in range(1,11):
                    rng=np.random.default_rng(seed)
                    xs=base.copy()
                    if layout=='whole_domain': xs=rbf.select_sensors(x,D,24,mode='all')
                    if layout=='jitter':
                        xs[1:-1]+=rng.uniform(-.35,.35,len(xs)-2)*np.median(np.diff(xs[1:-1]))
                        xs=np.unique(np.sort(np.clip(xs,x[0],x[-1])))
                    if layout.startswith('gap'):
                        fraction=int(layout[3:])/100
                        xs=xs[np.abs(xs-centre)>=fraction*width/2]
                    clean=np.interp(xs,x,D); noisy=clean.copy()
                    inside=(xs>x[0]) & (xs<x[-1])
                    noisy[inside]+=noise*max(np.max(np.abs(clean[inside])),1e-12)*rng.normal(size=inside.sum())
                    span=(float(xs[1]),float(xs[-2]))
                    candidates=[]
                    for nc in [max(len(xs)+2,8),max(len(xs)+6,10)]:
                        for wm in [.8,1.2]:
                            for ridge in [1e-8,1e-4]:
                                score=rbf.loo_score(xs,noisy,nc,wm,ridge,span)
                                if np.isfinite(score): candidates.append((score,nc,wm,ridge))
                    score,nc,wm,ridge=min(candidates)
                    estimate,*_=rbf.fit_rbf(x,xs,noisy,nc,wm,ridge,span=span)
                    pred=S-estimate/3
                    core=mask & (np.abs(x-centre)<.25*width)
                    records.append(dict(Mach=mach,layout=layout,noise=noise,seed=seed,
                      interior_count=int(inside.sum()),total_count=len(xs),
                      R_error=rbf.rel_l2(pred,R,mask),R_core_error=rbf.rel_l2(pred,R,core),
                      Delta_error=rbf.rel_l2(estimate,D,mask),ncentres=nc,width_mult=wm,ridge=ridge,loo=score))
                    sensors[f'M{mach}_{layout}_{noise}_{seed}']=xs.tolist()
        print('Sparse finished',mach,flush=True)
    for path in sorted((ORIG/'ref/mach_sweep').glob('*_fullstate.npz')):
        if not any(f'standing_M{m}_' in path.name for m in [2,3,5]): continue
        z=np.load(path); rho=z['rho']; T=z['T']; sigma=z['sig']
        eig=np.stack([T-1.25*sigma/rho,T+.625*sigma/rho,T+.625*sigma/rho],axis=1)
        np.savetxt(OUT/(path.stem+'_ESFP_eigenvalues.csv'),
                   np.column_stack([z['x'],eig]),delimiter=',',header='x,eig_x,eig_y,eig_z',comments='')
        admissibility.append(dict(file=str(path),minimum_diffusion_eigenvalue=float(eig.min()),
                                  nonpositive_cells=int(np.sum(eig.min(axis=1)<=0))))
    with open(OUT/'sparse_stress_tests.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    (OUT/'sensor_locations.json').write_text(json.dumps(sensors))
    (OUT/'ESFP_admissibility.json').write_text(json.dumps(admissibility,indent=2))
    (OUT/'DONE.json').write_text(json.dumps({'sparse_tests':len(records),
      'scope':'Existing-reference diagnostic; active-zone sensor selection uses reference Delta. Not a full ESFP solve or entropy-production proof.'}))

if __name__=='__main__': main()
