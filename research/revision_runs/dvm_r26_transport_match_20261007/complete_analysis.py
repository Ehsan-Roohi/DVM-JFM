from pathlib import Path
import json,hashlib,datetime,shutil,tarfile,subprocess
import numpy as np
B=Path('/scratch3/workspace/roohie_umass_edu-jfm1439-r26-m2-20261007');D=B/'dvm_transport_match';A=D/'completed_analysis_20261007';A.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fields=['rho','u','T','q','sigma','R','Delta','S'];manifest=json.loads((D/'manifest.json').read_text());profiles={};done={};archive_count=0
for c in manifest['cases']:
 out=D/'runs'/c['name'];z=json.loads((out/'DONE.json').read_text());assert z['config']==c and z['source']==sha(D/'solver.py')
 ar=json.loads((out/'ARCHIVED.json').read_text())
 for n,h in ar['sha256'].items():
  src=out/n if (out/n).exists() else D/n;assert sha(src)==h==sha(Path(ar['archive'])/n)
 archive_count+=len(ar['sha256']);a=np.load(out/'profiles.npz',allow_pickle=False);assert np.isfinite(a['fields']).all();profiles[c['name']]=(a['x'],a['fields']);done[c['name']]=z
assert archive_count==49
rpath=B/'r26_grid641/runs/M2_grid_N641/M2_N641_L30/profile.npz';a=np.load(rpath,allow_pickle=False);rfields=np.column_stack([a['theta'] if k=='T' else a[k] for k in fields]);assert np.isfinite(rfields).all();rx=a['x']
def align(x,f):
 mid=(1+16/7)/2;ii=np.flatnonzero((f[:-1,0]-mid)*(f[1:,0]-mid)<=0);assert len(ii)==1;i=ii[0];xs=x[i]+(mid-f[i,0])*(x[i+1]-x[i])/(f[i+1,0]-f[i,0]);return x-xs,f,float(xs)
xx=np.linspace(-15,15,1201)
def interp(x,f):
 assert x[0]<=xx[0] and x[-1]>=xx[-1];return np.column_stack([np.interp(xx,x,f[:,i]) for i in range(8)])
ax,rf,rs=align(rx,rfields);dx,df,ds=align(*profiles['M2_x1280']);yr=interp(ax,rf);yd=interp(dx,df)
comparison={k:{'relative_L2_percent_using_DVM_norm':float(100*np.linalg.norm(yr[:,i]-yd[:,i])/max(np.linalg.norm(yd[:,i]),1e-12)),'relative_Linf_percent_using_DVM_max':float(100*np.max(abs(yr[:,i]-yd[:,i]))/max(np.max(abs(yd[:,i])),1e-12))} for i,k in enumerate(fields)}
xb,fb,_=align(*profiles['M2_base']);yb=interp(xb,fb);sensitivity={}
for name in ['M2_v641','M2_vextent','M2_L45','M2_dt_half']:
 x,f,_=align(*profiles[name]);y=interp(x,f);sensitivity[name]={k:float(100*np.linalg.norm(y[:,i]-yb[:,i])/max(np.linalg.norm(yb[:,i]),1e-12)) for i,k in enumerate(fields)}
grid=json.loads((D/'grid_assessment/three_grid_report.json').read_text())
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_DVM_final_outputs_verified':True,'DVM_archive_hashes_verified':archive_count,'DVM_stationary_count':sum(z['stationary'] for z in done.values()),'DVM_finished_count':len(done),'DVM_final_diagnostics':done,'three_grid':grid,'terminal_time_sensitivity_percent':sensitivity,'R26_vs_Shakhov_DVM':{'R26_nodes':641,'DVM_cells':1280,'mach':2,'comparison_coordinate':'common physical length unit; density-midpoint translation only; x in[-15,15],1201samples; no width fitting','R26_midpoint':rs,'DVM_midpoint':ds,'metrics':comparison,'same_collision_operator':False,'validation_claim':False},'limitations':['All DVM profiles reached final time400 without unchanged stationarity<1e-7 for5samples.','DVM spatial grids also change CFL time step; use separate half-step sensitivity.','Shakhov and Maxwell R26 share Mach2,Pr2/3,mu=theta,farfields and length unit; complete collision operators differ.','Observed grid orders and finite-time differences are diagnostics, not certified solution error or full physical acceptance.','Signed reduced distributions have small negative fractions; nonnegative distribution not validated.'],'source_hashes':{'DVM_solver':sha(D/'solver.py'),'DVM_manifest':sha(D/'manifest.json'),'R26_fine_profile':sha(rpath)}}
(A/'analysis.json').write_text(json.dumps(r,indent=2));shutil.copy2(D/'grid_assessment/three_grid_report.json',A/'three_grid_report.json')
report='گزارش نهایی محاسبات ماخ ۲ با شرایط انتقال همسان — ۷ اکتبر ۲۰۲۶\nهفت کیس DVM تمام و ۴۹ هش آرشیو مستقل تأیید شده‌اند؛ هیچ کیس DVM ایستا طبق معیار اصلی نیست.\nسه‌شبکه R26 برای R/Delta/S اختلاف متوسط–ریز برابر ۰٫۱۰۷۷۳/۰٫۰۸۶۰۶/۰٫۰۹۶۹۶ درصد و مرتبه مشاهده‌شده نزدیک ۲ دارد.\nسه‌شبکه DVM برای R/Delta/S اختلاف متوسط–ریز برابر ۱٫۹۸۵۹۴/۱٫۸۴۶۱۰/۱٫۹۳۱۲۱ درصد و مرتبه مشاهده‌شده نزدیک ۰٫۹۵ دارد؛ با upwind مرتبه اول سازگار است، اما استقلال شبکهٔ ممان‌های چهارم اثبات نشده است. اختلاف چگالی ۰٫۱۱۶۴۸ درصد است.\nبا ریزکردن شبکهٔ مکانی، گام زمانی CFL نیز کوچک می‌شود. آزمون مستقل نصف‌کردن گام زمانی و تغییر شبکه و دامنهٔ سرعت در analysis.json آمده است.\nمقایسهٔ R26 شبکهٔ ۶۴۱ و DVM شبکهٔ ۱۲۸۰ روی طول فیزیکی یکسان و با هم‌ترازی مرکز چگالی انجام شده است؛ عرض شاک فیت نشده است.\n'
report+='اختلاف L2 مدل‌ها نسبت به نرم DVM (درصد): '+', '.join(k+'='+format(v['relative_L2_percent_using_DVM_norm'],'.5f') for k,v in comparison.items())+'\n'
report+='این اختلاف‌ها خطای صرف R26 نیستند: عملگرهای برخورد متفاوت‌اند و ایستایی DVM هنوز پاس نشده است. کسرهای منفی توزیع و شارهای وجه در داده‌ها ثبت شده‌اند. هیچ آستانه یا فیزیکی تغییر نکرده است.\n'
(A/'report_FA.txt').write_text(report,encoding='utf-8')
for c in manifest['cases']:
 out=D/'runs'/c['name'];target=A/'DVM_profiles'/c['name'];target.mkdir(parents=True,exist_ok=True)
 for n in ['profiles.npz','DONE.json','ARCHIVED.json']:shutil.copy2(out/n,target/n)
for n in ['solver.py','manifest.json','PASS.json']:shutil.copy2(D/n,A/n)
shutil.copy2(rpath,A/'R26_M2_N641_profile.npz')
source_files=[p for p in A.rglob('*') if p.is_file() and p.name not in ['bundle.tar.gz','ARCHIVE_RECEIPT.json']]
archive=Path('/project/pi_roohie_umass_edu/BGK_shock/revision_20260929_gpu_preempt/dvm_r26_transport_match_completed_analysis_20261007');archive.mkdir(parents=True,exist_ok=True);hashes={}
for src in source_files:
 rel=src.relative_to(A);dst=archive/rel;dst.parent.mkdir(parents=True,exist_ok=True);h=sha(src)
 if dst.exists():assert sha(dst)==h
 else:shutil.copy2(src,dst)
 assert sha(dst)==h;hashes[str(rel)]=h
receipt={'utc':r['utc'],'archive':str(archive),'source':str(A),'files_verified':len(hashes),'sha256':hashes,'DVM_final_output_archive_hashes_verified':49};(A/'ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2));shutil.copy2(A/'ARCHIVE_RECEIPT.json',archive/'ARCHIVE_RECEIPT.json')
with tarfile.open(A/'bundle.tar.gz','w:gz') as t:
 for src in source_files+[A/'ARCHIVE_RECEIPT.json']:t.add(src,arcname=str(src.relative_to(A)))
shutil.copy2(A/'bundle.tar.gz',archive/'bundle.tar.gz');assert sha(A/'bundle.tar.gz')==sha(archive/'bundle.tar.gz')
print(json.dumps({'utc':r['utc'],'files_verified':len(hashes),'archive':str(archive),'bundle':str(A/'bundle.tar.gz'),'bundle_sha256':sha(A/'bundle.tar.gz'),'DVM_archive_hashes':49,'comparison':comparison,'DVM_stationary_count':0},indent=2))
