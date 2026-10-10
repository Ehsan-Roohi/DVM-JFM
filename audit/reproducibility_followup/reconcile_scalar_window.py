"""Cheap reconciliation of full-f reductions with actual archived scalar window.
Does not reload distribution or recompute target. Original RESULT.json remains
immutable: its window1 comparison is superseded by this explicit window7 audit.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):
    rows=list(csv.DictReader(p.open()));return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
def avg(y,n):return np.convolve(np.pad(y,(n//2,n//2),mode='edge'),np.ones(n)/n,mode='valid')
def plateau(y):
    n=max(5,int(.08*len(y)));return y-np.mean(np.r_[y[:n],y[-n:]])
def rel(a,b,m):return float(np.linalg.norm((a-b)[m])/np.linalg.norm(b[m]))
def corr(a,b,m):return float(np.corrcoef(a[m],b[m])[0,1])
def evidence():
    for p in (HERE,*HERE.parents):
        for d in [p/'research/revision_runs/model_audit/evidence',p/'publication/research/revision_runs/model_audit/evidence']:
            if (d/'5.csv').exists():return d
    raise FileNotFoundError('Evidence CSVs missing')
def run(mach):
    directory=HERE/f'BGK_M{mach}_regenerated';z=np.load(directory/'reductions.npz');x=z['x']
    scalar=read(HERE/f'M{mach}_delta_budget.profiles.csv');heat=read(evidence()/f'{ {2:5,3:6,5:7}[mach]}.csv')
    windows={};J=z['stored_centre_J'];K=z['stored_centre_K'];u=heat['ux_file']
    for n in (1,3,5,7,9,11):
        windows[str(n)]={'dJdx_max':float(np.max(abs(np.gradient(avg(J,n),x)-scalar['dJdx']))),
            'dudx_max':float(np.max(abs(np.gradient(avg(u,n),x)-scalar['dudx'])))}
    n=7;forcing=np.gradient(avg(J,n),x)+4*np.gradient(avg(u,n),x)*avg(K,n)
    reproduced=plateau(-forcing)
    reproduction={'forcing_max_abs':float(np.max(abs(forcing-scalar['forcing']))),
        'Delta_rec_max_abs':float(np.max(abs(reproduced-scalar['Delta_rec_from_budget']))),
        'Delta_ref_max_abs':float(np.max(abs(plateau(z['stored_centre_D'])-scalar['Delta'])))}
    # Full-f reductions must match archived scalar generation, not merely agree qualitatively.
    assert reproduction['forcing_max_abs']<1e-8
    assert reproduction['Delta_rec_max_abs']<1e-8
    results={};target_errors={};out={'x':x,'Delta_original_window7':reproduced}
    legacy=plateau(heat['S_R_plus_Delta3_rec']);fixed=abs(legacy)>=.02*np.max(abs(legacy))
    for name in ('stored_centre','recomputed_macros'):
        p=name+'_';u=heat['ux_file'] if name=='stored_centre' else z[p+'u']
        rho=heat['rho_file'] if name=='stored_centre' else z[p+'rho']
        T=heat['T_file'] if name=='stored_centre' else z[p+'T_at_centre']
        target_errors[name]={'mass_max_abs':float(np.max(abs(z[p+'rhoMd']-rho))),
            'mean_max_abs':float(np.max(abs(z[p+'uMd']-u))),
            'temperature_max_abs':float(np.max(abs(z[p+'TMd']-T))),
            'target_arithmetic':'float64 four-iteration on actual stored grid/weights; not original GPU identity'}
        force=np.gradient(avg(z[p+'J'],7),x)+4*np.gradient(avg(u,7),x)*avg(z[p+'K'],7)
        Drec=z[p+'C4Md']-force-15*rho*T*T
        D=plateau(z[p+'D']);R=plateau(z[p+'R']);S=plateau(z[p+'S_rec']);Db=plateau(Drec)
        newmask=abs(S)>=.02*np.max(abs(S));results[name]={}
        for label,m in [('fixed_legacy',fixed),('corrected_reconstructed_S',newmask)]:
            results[name][label]={'active_points':int(m.sum()),'Delta_budget_relL2':rel(Db,D,m),
                'R_delta0_relL2':rel(S,R,m),'R_fullDelta_relL2':rel(S-D/3,R,m),'R_budgetDelta_relL2':rel(S-Db/3,R,m),
                'Delta_budget_correlation':corr(Db,D,m),'R_budgetDelta_correlation':corr(S-Db/3,R,m)}
        out[p+'D_rec_window7']=Drec;out[p+'scalar_forcing_window7']=force
    report={'status':'PASS_FULLSTATE_SCALAR_WINDOW7_REPRODUCTION','Mach':mach,'actual_archived_smoothing_window':7,
        'window_tests':windows,'original_scalar_reproduction':reproduction,'corrected_window7_table_metrics':results,
        'supersedes':'RESULT.json scalar window1 mismatch/table entries only; full-f reductions unchanged',
        'limitations':'Actual finite-time profile; differential-budget derivative7 and plateau2% convention retained; double target reconstruction, no evolution/stationarity.'}
    np.savez(directory/'scalar_window7_reconciliation.npz',**out)
    (directory/'RESULT_scalar_window7.json').write_text(json.dumps(report,indent=2)+'\n')
    original=json.loads((directory/'RESULT.json').read_text())
    contract={'status':'FULLSTATE_ARCHIVED_CONTRACT_RECONCILED_WINDOW7','Mach':mach,
        'input_sha256':original['input_sha256'],'distribution_npy_sha256':original['f_npy_sha256'],
        'heat_CSV_reproduction':original['heat_original_CSV_reproduction'],
        'scalar_CSV_reproduction':reproduction,'legacy_table_exact_mask_metrics':original['original_table_mask_metrics'],
        'corrected_table_metrics_window7':results,'target_moment_errors_vs_requested_macros':target_errors,
        'source_and_anchor_diagnostics':{name:{key:value for key,value in details.items() if key.startswith(('anchor_','source_','qMd_','DeltaMd_'))} for name,details in original['corrected_outcomes'].items()},
        'contract':{'heat_derivative_smoothing':'one binomial1,4,6,4,1 pass on u only','scalar_derivative_smoothing':'boxcar7 on J,u,K','coordinate':'x_mfp','tau':1,'plateau':'mean first/last8% removed','fixed_legacy_mask':'abs(plateau(originalSrec))>=.02peak','corrected_mask':'abs(plateau(correctedSrec))>=.02peak'},
        'supersedes':'Original RESULT.json scalar-window1 table outputs are unqualified historical diagnostics; use this contract and scalar_window7_reconciliation.npz',
        'limitations':report['limitations']}
    (directory/'CONTRACT_RECONCILIATION.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps({'Mach':mach,'reproduction':reproduction,'stored_centre_fixedmask':results['stored_centre']['fixed_legacy']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mach',type=int,required=True,choices=[2,3,5]);a=p.parse_args();run(a.mach)
