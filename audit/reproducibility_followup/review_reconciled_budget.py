"""Independent small-array audit of archived/corrected table contracts; no full-f load."""
import csv,json,hashlib,argparse
from pathlib import Path
import numpy as np
F=Path(__file__).resolve().parent
def read(p):
    with p.open() as f:rows=list(csv.DictReader(f))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
def zero(v):
    ne=max(5,int(.08*len(v)));return v-(v[:ne].sum()+v[-ne:].sum())/(2*ne)
def avg7(v):return np.convolve(np.pad(v,(3,3),mode='edge'),np.full(7,1/7),mode='valid')
def norm(a,b,m):return float(np.sqrt(np.sum((a[m]-b[m])**2)/np.sum(b[m]**2)))
def main(mach):
    d=F/f'BGK_M{mach}_regenerated';report=json.loads((d/'CONTRACT_RECONCILIATION.json').read_text());z=np.load(d/'reductions.npz');w=np.load(d/'scalar_window7_reconciliation.npz')
    hist=F.parent/'publication/research/revision_runs/model_audit/evidence'/f'{ {2:5,3:6,5:7}[mach]}.csv';h=read(hist);s=read(F/f'M{mach}_delta_budget.profiles.csv');x=z['x']
    S0=zero(h['S_R_plus_Delta3_rec']);R0=zero(h['Rxx_closure']);D0=zero(h['Delta']);Db0=zero(s['Delta_rec_from_budget']);mask=np.abs(S0)>=.02*np.max(np.abs(S0))
    force=np.gradient(avg7(z['stored_centre_J']),x)+4*np.gradient(avg7(h['ux_file']),x)*avg7(z['stored_centre_K'])
    forcing_difference=float(np.max(np.abs(force-s['forcing'])));assert forcing_difference<1e-8
    delta_original_difference=float(np.max(np.abs(zero(-force)-s['Delta_rec_from_budget'])));assert delta_original_difference<1e-8
    baseline={'R_budgetDelta_relL2':norm(S0-Db0/3,R0,mask),'Delta_budget_relL2':norm(Db0,D0,mask)}
    for k,v in baseline.items():assert abs(v-report['legacy_table_exact_mask_metrics'][k])<1e-12
    branches={}
    for branch in ['stored_centre','recomputed_macros']:
        pre=branch+'_';u=h['ux_file'] if branch=='stored_centre' else z[pre+'u'];rho=h['rho_file'] if branch=='stored_centre' else z[pre+'rho'];T=h['T_file'] if branch=='stored_centre' else z[pre+'T_at_centre']
        Q=np.gradient(avg7(z[pre+'J']),x)+4*np.gradient(avg7(u),x)*avg7(z[pre+'K'])
        delta_new=z[pre+'C4Md']-Q-15*rho*T*T
        assert np.max(abs(delta_new-w[pre+'D_rec_window7']))<1e-11
        delta_md=z[pre+'C4Md']-15*rho*T*T
        # Exact decomposition: new scalar reconstruction = old forcing + Md excess.
        assert np.max(abs(zero(delta_new)-(zero(-Q)+zero(delta_md))))<1e-11
        S=zero(z[pre+'S_rec']);R=zero(z[pre+'R']);D=zero(z[pre+'D']);Db=zero(delta_new)
        metric={'R_budgetDelta_relL2':norm(S-Db/3,R,mask),'Delta_budget_relL2':norm(Db,D,mask)}
        for k,v in metric.items():assert abs(v-report['corrected_table_metrics_window7'][branch]['fixed_legacy'][k])<1e-12
        unchangedmask=np.array_equal(mask,abs(S)>=.02*np.max(abs(S)))
        branches[branch]={'independently_computed':metric,'fixed_mask_points':int(mask.sum()),'corrected_mask_identical':unchangedmask,'Md_scalar_excess_max':float(np.max(abs(delta_md))),'Md_excess_plateau_max':float(np.max(abs(zero(delta_md)))),'R_error_without_Md_excess_same_corrected_heat':norm(S-zero(-Q)/3,R,mask),'Md_constraint_errors':{'mass':float(np.max(abs(z[pre+'rhoMd']-rho))),'mean':float(np.max(abs(z[pre+'uMd']-u))),'temperature':float(np.max(abs(z[pre+'TMd']-T)))}}
    result={'status':'PASS_INDEPENDENT_SMALL_ARRAY_CONTRACT_REVIEW','Mach':mach,'source_review':'Geometry A and scalar forcing signs correct; exact archived smoothing uses boxcar7(J,u,K), heat binomial(u), plateau8% andfixed2%mask.','baseline':baseline,'scalar_original_forcing_max_abs':forcing_difference,'scalar_original_Delta_max_abs':delta_original_difference,'branches':branches,'scope':'Independent arithmetic from preserved compact full-f reductions; no distribution/target rerun and no steady/convergence claim.'}
    (d/'INDEPENDENT_CONTRACT_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mach',type=int,required=True,choices=[2,3,5]);main(p.parse_args().mach)
