from pathlib import Path
import numpy as np, json, hashlib, csv
HERE=Path(__file__).resolve().parent
E=HERE/'M2_raw'; OUT=HERE
paths={'DSMC_2000ppc':E/'DSMC_2000ppc_M2.npz','Shakhov_DVM':E/'DVM_Shakhov_M2.npz','Corrected_Shakhov_R26':E/'R26_accepted_Shakhov_M2.npz'}
data={}
for name,p in paths.items():
 a=np.load(p,allow_pickle=False); d={k:a[k] for k in a.files}
 if name=='Shakhov_DVM': d.update({str(k):a['fields'][:,i] for i,k in enumerate(a['columns'])})
 if 'theta' in d: d['T']=d['theta']
 data[name]=d
fields=['rho','T','q','R','Delta','S']; target=(1+16/7)/2; centers={}
for n,d in data.items():
 x=d['x']; y=d['rho']-target; ii=np.where(y[:-1]*y[1:]<=0)[0]
 if not len(ii): raise ValueError(f'No density-midpoint crossing for {n}')
 crossings=[float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])) for i in ii]
 center=min(crossings,key=abs); centers[n]={'x':center,'crossing_count':len(ii)}; d['xc']=x-center
x=np.linspace(-15,15,1201)
aligned={n:{f:np.interp(x,d['xc'],d[f]) for f in fields} for n,d in data.items()}
metrics={n:{f:float(100*np.linalg.norm(aligned[n][f]-aligned['DSMC_2000ppc'][f])/np.linalg.norm(aligned['DSMC_2000ppc'][f])) for f in fields} for n in ['Shakhov_DVM','Corrected_Shakhov_R26']}
channel={}
ref=aligned['DSMC_2000ppc']
for n in ['Shakhov_DVM','Corrected_Shakhov_R26']:
 e_r=aligned[n]['R']-ref['R']; e_d3=(aligned[n]['Delta']-ref['Delta'])/3; e_s=aligned[n]['S']-ref['S']
 nr,nd,ns=(np.linalg.norm(a) for a in (e_r,e_d3,e_s)); scale=np.linalg.norm(ref['S'])
 assert np.max(np.abs(e_s-e_r-e_d3))<1e-10
 channel[n]={'error_R_percent_common_S_denominator':float(100*nr/scale),'error_Delta_over_3_percent_common_S_denominator':float(100*nd/scale),'error_S_percent_common_S_denominator':float(100*ns/scale),'signed_cross_correlation':float(np.dot(e_r,e_d3)/(nr*nd)),'cancellation_ratio':float(ns/(nr+nd))}
result={'method':'100*L2(model-DSMC2000)/L2(DSMC2000); common x in [-15,15] lambda1, 1201 uniform interpolation points; Rankine-Hugoniot density midpoint translation only; no width fit or smoothing. Grid points are not independent samples.','M':2,'RH_density_midpoint':target,'centers':centers,'relative_L2_percent_DSMC_denominator':metrics,'fields':fields,'input_sha256':{n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()},'notes':['R26 input is corrected consistent Shakhov p4/q8 profile; exclude historical pre-STF Maxwellian R26 profile.','DVM values use DSMC denominator; older numbers in DSMC_M2_M5_report.json were not suitable for this table because its R26 row referred to the historical profile and its DVM norm denominator differed from the caption.','Finite-time, single-seed DSMC; DVM is a finite-time snapshot; pairwise differences are descriptive, not certified model error.']}
result['channel_signed_error_common_DSMC_S_denominator']=channel
(OUT/'DSMC2000_M2_corrected_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (OUT/'DSMC2000_M2_corrected_comparison.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['model',*fields]);
 for n in metrics:w.writerow([n,*[metrics[n][k] for k in fields]])
print(json.dumps(result,indent=2))


