from pathlib import Path
import json,datetime,numpy as np,hashlib
R=Path('revision_runs');P=R/'scientific_audit_20261002'
z=np.load(P/'velocity_refine_final.npz');b=np.load('paper_figures/reviewer_revision/inputs/M5_baseline.npz');g=np.linspace(-15,15,601)
def align(z,k):
 x=z['x_mfp'];r=z['rho'];mid=(r[0]+r[-1])/2;i=np.where((r[:-1]<mid)&(r[1:]>=mid))[0][0];c=x[i]+(mid-r[i])*(x[i+1]-x[i])/(r[i+1]-r[i]);return np.interp(g+c,x,z[k])
metrics={k:float(100*np.linalg.norm(align(b,k)-align(z,k))/np.linalg.norm(align(z,k))) for k in ['rho','T','qx','Rxx_closure','Delta_dvm','S_dvm']}
(P/'final_velocity_comparison.json').write_text(json.dumps({'relative_L2_percent':metrics,'denominator':'refined final profile','steps':52000,'sha256':hashlib.sha256((P/'velocity_refine_final.npz').read_bytes()).hexdigest()},indent=2))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report=f'''گزارش چهار بررسی علمی — {now}

۱. DVM و دنباله توزیع
هر ۶۲ کیس اصلی اکنون از نظر خروجی معتبر کامل است. آخرین شبکه سرعت در گام ۵۲۰۰۰: checkpoint متناهی، تنظیمات و هش منبع/gate مطابق و ۳۵ فایل آرشیو در هر دو مسیر مستقل SHA256 تأیید شد. تخصیص منتظر65157835_2 پس از این تأیید لغو شد تا کار کامل تکرار نشود.
اختلاف L2 نهایی نسبت به baseline، با مخرج پروفایل ریزتر (%): {json.dumps(metrics)}.
در ردیف‌های انتخاب‌شده، سهم c^4 در ده درصد بیرونی جعبه سرعت حدود3e-8 یا کمتر بود؛ این سهم حل‌شده است و کران قطعی دنباله خارج از جعبه نیست.

۲. پایلوت DSMC اصلاح‌شده
دو seed در PPC960 اختلاف R/Delta/S حدود1.013/0.886/0.772 درصد و منبع برخورد2.049 درصد دارند. دو seed برای تفکیک اثر الگوریتم از نویز کافی نیست.
۱۲ اجرای اصلاح‌شده برای seed3..8 و PPC120/960 ثبت شد؛ کنترل‌های اصلی موجود دوباره اجرا نمی‌شوند. آرایه65158941، gate65158940 با خروج موفق و هش‌های مستقل تأییدشده؛ پنج task در حال اجرا و task5 منتظر در آخرین مشاهده. gpu-preempt، بدون throttle، scratch موجود با انقضای31اکتبر. این۱۲ اجرا خارج از شمار62کیس اصلی‌اند.
آرایه اولیه65158908 پیش از محاسبه با AttributeError در sha(__file__) شکست خورد. Path(p) اصلاح شد، تست str/Path گذشت و آزمون تازه parity/restart و10000برخورد اجرا شد. هیچ تغییری در حلگر علمی اعمال نشد. شکست‌های اولیه ثبت sbatch ناشی از CRLF و dependency منقضی نیز job تولید نکردند.

۳. نمونه‌برداری زمانی و clipping شاخوف
در8seed اصلی PPC3840، اختلاف نیمه اول/دوم میانگین ممان‌های R/Delta/S برابر0.285/0.262/0.193 درصد و منبع برخورد1.034 درصد بود. ده بلوک برای اثبات استقلال کامل یا ESS کافی نیست. فاصله bootstrap نرم نامنفی تشخیصی است؛ آزمون صفر بودن drift نیست.
آزمون کوتاه20sweep شاخوف نشان می‌دهد clipping جرم و انرژی غیرصفر اضافه می‌کند. بیشینه افزایش تک‌رویدادی نرمال‌شده جرم4.695e-5، انرژی1.327e-3، ممان چهارم7.889e-3 است. جمع بیشینه‌های ردیفی کران بالا است، نه خطای انباشته در یک سلول و نه نتیجه حالت پایا. مسئله هنوز حل علمی نشده است.

۴. آنتروپی ES-FP
اپراتور محلی ES-FP با beta=-1.25 روی توزیع‌های مرجع BGK ارزیابی شد؛ این حل شوک ES-FP نیست. دو تست تحلیلی ماکسولی/گاوسی گذشت. در هسته شوک مقدار محاسبه‌شده مثبت بود، اما حساسیت مشتق مرتبه2/4 از15.93 درصد شبکه پایه به9.17 درصد شبکه ریز کاهش یافت و هنوز کوچک نیست. بیشینه باقیمانده انرژی نرمال‌شده از0.103 به0.01995 کاهش یافت.
آنتروپی منفی دوردست شبکه درشت با کنترل ماکسولی همان شبکه نیز بازتولید می‌شود؛ نمی‌توان آن را نقض فیزیکی H-theorem دانست. هم‌زمان، این تشخیص عددی تأیید H-theorem هم نیست. نتیجه قطعی آنتروپی نیازمند دقت بیشتر/اپراتور گسسته سازگار است.

فایل‌های JSON روش، داده و provenance را نگه می‌دارند. scientific_diagnostics.png/svg نمودار تشخیصی است. فعلاً نتیجه آنتروپی و پایلوت دوseed نباید به ادعای تأیید نهایی مقاله تبدیل شود. گزارش تفصیلی مناسب اپندیکس است؛ محدودیت clipping و مرز اعتبار ادعا باید در متن اصلی هم بیان شود.
'''
(P/'report_FA.txt').write_text(report,encoding='utf-8')
s=json.loads((R/'monitor_state.json').read_text(encoding='utf-8-sig'))
e={'at_utc':now,'event':'four_scientific_audits','validated_completed':62,'report':str(P/'report_FA.txt'),'new_gate':'65158940','new_array':'65158941','replaced_array':'65158908','diagnosis':'runner sha(str) type error; fixed Path conversion; fresh verified gate','extra_scope':'12 corrected DSMC missing seeds3-8 PPC120/960; distinct from62 main cases','remaining':'extra ensemble, hardware gates, entropy accuracy and Shakhov clipping scientific assessment'}
s['validated_completed']=62;s['observed_at_utc']=now;s['scientific_audit_20261002']=e;s.setdefault('events',[]).append(e)
s['corrected_dsmc_extra']=json.loads((R/'dsmc_pairfix/extra_repair_20261002.json').read_text());s['corrected_dsmc_extra']['replaces']='65158908';s['corrected_dsmc_extra']['gate_verified']=True
for run in s['runs']:
 if run.get('name')=='M5_velocity_refine':run.update(state='COMPLETED',validated=True,checkpoint_step=52000,archive_verified_files=35,validation_report=str(P/'final_dvm_validation.json'))
(R/'monitor_state.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')
with (R/'monitor_report.txt').open('a',encoding='utf-8') as f:f.write('\n\n'+report)
j=json.loads((R/'submitted_jobs.json').read_text(encoding='utf-8-sig'));j['corrected_dsmc_extra_20261002']=e;(R/'submitted_jobs.json').write_text(json.dumps(j,indent=2,ensure_ascii=False),encoding='utf-8')
print(metrics)
