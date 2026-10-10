from pathlib import Path
import subprocess,sys,json,importlib.util,ast
import numpy as np,pandas as pd
r=Path('revision_runs/model_audit');o=r/'corrected_20261001'
def load(name):
 spec=importlib.util.spec_from_file_location(name,o/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('plot_shakhov_channel_consistency_final');d=pd.read_csv(o/'inputs/M3.profiles.csv');tree=ast.parse((o/'plot_shakhov_channel_consistency_final.py').read_text());assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='Dref_col' for t in n.targets));selected=eval(compile(ast.Expression(assignment.value),'<test>','eval'),{'pick':m.pick,'df':d});assert selected=='Delta'
try:eval(compile(ast.Expression(assignment.value),'<test>','eval'),{'pick':m.pick,'df':pd.DataFrame({'Delta_target':[0.]})});raise AssertionError('Accepted target-only data')
except KeyError:pass
out={'shakhov_physical_reference':selected,'target_only_rejected':True,'ESFP':{}}
for mach,idx in [('M2',5),('M3',6),('M5',7)]:
 subprocess.run([sys.executable,str(o/'esfp_scalar_excess_source_from_profiles.py'),'--profiles',str(r/f'evidence/{idx}.csv'),'--case',mach,'--outdir',str(o/'esfp'),'--Kn','1'],check=True,stdout=subprocess.DEVNULL)
 metric=next((o/'esfp').rglob(mach+'*metrics.csv'));z=pd.read_csv(metric).iloc[0]
 a=float(z['ESFP_corr_over_Delta_active']);b=float(z['ESFP_corr_over_Delta_active_direct']);assert np.isfinite(a) and a==b
 source=pd.read_csv(r/f'evidence/{idx}.csv');mask=np.abs(source.Delta)>0.02*np.abs(source.Delta).max();rho=source['rho_from_f'] if 'rho_from_f' in source else source['rho'];expected=np.linalg.norm((-3.75*source.sigma_xx**2/rho)[mask])/np.linalg.norm(source.Delta[mask]);np.testing.assert_allclose(a,expected,rtol=1e-12)
 out['ESFP'][mach]=a
m=load('plot_two_budget_recovery_jfm');tree=ast.parse((o/'plot_two_budget_recovery_jfm.py').read_text());assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and any(isinstance(t,ast.Name) and t.id=='sname' for t in n.targets[0].elts));expr=compile(ast.Expression(assign.value),'<test>','eval')
for idx in [5,6,7]:
 d=pd.read_csv(r/f'evidence/{idx}.csv');a,name=eval(expr,{'pick':m.pick,'h':d});assert name=='S_R_plus_Delta3_rec'
 try:eval(expr,{'pick':m.pick,'h':d.drop(columns=[name])});raise AssertionError('Accepted missing reconstruction')
 except KeyError:pass
out['independent_reconstruction_required']='PASS M2/M3/M5; reference fallback rejected'
(o/'regression_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
