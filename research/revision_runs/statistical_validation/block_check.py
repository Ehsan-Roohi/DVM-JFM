from pathlib import Path
import numpy as np,json
p=Path('/scratch4/workspace/roohie_umass_edu-jfm1439-ppc1920-3840-20261001/outputs/dsmc')
fields=['n','T','Rxx_closure','Delta','S'];out={'method':'Ten nonoverlapping blocks80000:180000 derived by differencing cumulative sample sums;1000 records per block. Excludes earliest1001records. Independent seeds8. Lag1 across only10blocks is descriptive, not proof of independence or an effective sample size.','seeds':{}}
for seed in range(1,9):
 d=p/f'ppc3840_seed{seed}';snap=[np.genfromtxt(d/f'M3_step{i:08d}.csv',delimiter=',',names=True) for i in range(80000,180001,10000)];blocks=[]
 for a,b in zip(snap[:-1],snap[1:]):
  na=a['nsamples'];nb=b['nsamples'];assert np.all(nb-na==1000)
  v={k:(nb*b[k]-na*a[k])/(nb-na) for k in fields};assert all(np.isfinite(w).all() for w in v.values());blocks.append(v)
 x=snap[0]['x'];aligned={k:[] for k in fields};centers=[]
 for b in blocks:
  idx=np.where((b['n'][:-1]<2)&(b['n'][1:]>=2))[0];assert len(idx)==1;i=idx[0];c=x[i]+(2-b['n'][i])*(x[i+1]-x[i])/(b['n'][i+1]-b['n'][i]);centers.append(float(c))
  for k in fields:aligned[k].append(np.interp(x+c,x,b[k],left=np.nan,right=np.nan))
 mask=np.abs(x)<=15;metrics={}
 for k in fields:
  v=np.array(aligned[k])[:,mask];first=v[:5].mean(0);last=v[5:].mean(0);strength=np.linalg.norm(v,axis=1);lag=float(np.corrcoef(strength[:-1],strength[1:])[0,1]);metrics[k]={'first_vs_last_half_L2_percent':float(100*np.linalg.norm(first-last)/np.linalg.norm(last)),'block_norm_lag1_descriptive':lag,'block_norms':strength.tolist()}
 out['seeds'][str(seed)]={'centers':centers,'metrics':metrics}
print(json.dumps(out))
