"""Read-only CPU/CUDA Figure5 comparison; no solver/target recomputation."""
from pathlib import Path
import argparse,csv,hashlib,json,math

F=Path(__file__).resolve().parent
A=F.parent
def evidence_root():
    for root in [F,*F.parents]:
        for rel in ['research/revision_runs/model_audit/evidence','publication/research/revision_runs/model_audit/evidence']:
            candidate=root/rel
            if (candidate/'2.csv').is_file() and (candidate/'3.csv').is_file():return candidate
    raise FileNotFoundError('Find repository research/revision_runs/model_audit/evidence or audit publication equivalent')
H=evidence_root()
EXPECTED={'historical_profile':'c0fa17cc456bd40b5227fa710a36080fcc8473315f20e7c5002bd58e29971a91','historical_metrics':'29b46b616e1ebbc2a6bde3ca42e22a335cac58ddf211f43378cbe197ff00c1f1','fullstate':'9077c010ac9169e573e8b47c55864496f220ea9f5c5f517b08ac90603fb45011'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def numeric(rows,k):
    try:return [float(r[k]) for r in rows]
    except (ValueError,KeyError):return None
def rms(v):return math.sqrt(math.fsum(x*x for x in v)/len(v))
def compare_columns(old,new):
    assert len(old)==len(new),('row_count',len(old),len(new))
    assert old[0].keys()==new[0].keys(),('column_set',list(old[0]),list(new[0]))
    out={}
    for k in old[0]:
        x=numeric(old,k);y=numeric(new,k)
        if x is None or y is None:
            out[k]={'numeric':False,'identical':all(a[k]==b[k] for a,b in zip(old,new))};continue
        assert all(math.isfinite(v) for v in x+y),('nonfinite',k)
        d=[b-a for a,b in zip(x,y)];scale=max(max(abs(v) for v in x),max(abs(v) for v in y),1e-30)
        out[k]={'numeric':True,'max_abs':max(abs(v) for v in d),'rms_abs':rms(d),'scale_max':scale,'max_over_scale':max(abs(v) for v in d)/scale,'relative_L2':rms(d)/max(rms(x),1e-30),'reference_min':min(x),'reference_max':max(x),'new_min':min(y),'new_max':max(y)}
    return out
def rel(a,b,m):return math.sqrt(math.fsum((x-y)**2 for x,y,q in zip(a,b,m) if q)/max(math.fsum(y*y for y,q in zip(b,m) if q),1e-300))
def metrics_check(rows,met):
    keys={'S_flux_relL2_active':('S_from_flux','S'),'Delta_source_relL2_active':('Delta_from_source','Delta'),'R_delta0_relL2_active':('R_delta0','R'),'R_two_channel_relL2_active':('R_two_channel','R')}
    mask=[bool(int(r['active_mask'])) for r in rows];out={}
    for k,(a,b) in keys.items():
        v=rel(numeric(rows,a),numeric(rows,b),mask);stored=float(met[0][k]);out[k]={'recomputed':v,'stored':stored,'absolute_difference':abs(v-stored),'CSV_internal_match':abs(v-stored)<=1e-12+1e-10*abs(stored)}
    q=numeric(rows,'qx');Q=numeric(rows,'Qq_direct');v=math.fsum(a/(b+1e-300) for a,b,m in zip(Q,q,mask) if m)/sum(mask)
    out['mean_Qq_over_q_active']={'recomputed':v,'stored':float(met[0]['mean_Qq_over_q_active']),'absolute_difference':abs(v-float(met[0]['mean_Qq_over_q_active']))}
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--regenerated',type=Path,default=F/'figure5_regenerated');ap.add_argument('--fullstate',type=Path);ap.add_argument('--output',type=Path,default=F/'FIGURE5_CPU_REGEN_COMPARISON.json');args=ap.parse_args()
    ps=list(args.regenerated.rglob('*.profiles.csv'));ms=list(args.regenerated.rglob('*.metrics.csv'))
    assert len(ps)==len(ms)==1,('need_exactly_one_regenerated_pair',ps,ms)
    oldp=H/'3.csv';oldm=H/'2.csv';assert sha(oldp)==EXPECTED['historical_profile'];assert sha(oldm)==EXPECTED['historical_metrics']
    old=read(oldp);new=read(ps[0]);om=read(oldm);nm=read(ms[0]);columns=compare_columns(old,new);metrics=compare_columns(om,nm)
    # Prospectively explicit tolerance: physical columns on a common peak scale,
    # not pointwise relative errors at zero crossings. Tiny cancellation metrics
    # are reported in full and never given an artificial relative-match pass.
    principal=['rho','ux','T','qx','sigma_xx','R','Delta','S','S_from_flux','Delta_from_source','R_delta0','R_two_channel']
    principal_pass=all(columns[k]['relative_L2']<=1e-4 and columns[k]['max_over_scale']<=5e-4 for k in principal)
    structural_pass=columns['x']['max_abs']==0 and columns['active_mask']['max_abs']==0
    internal_old=metrics_check(old,om);internal_new=metrics_check(new,nm)
    internal_pass=all(v.get('CSV_internal_match',True) for g in [internal_old,internal_new] for v in g.values())
    fullstate=None
    if args.fullstate:
        fullstate={'path':str(args.fullstate),'bytes':args.fullstate.stat().st_size,'sha256':sha(args.fullstate)};assert fullstate['sha256']==EXPECTED['fullstate']
    report={'audit':'fullstate-derived CPU output vs historical CUDA sourcefixed CSV; no numerical solver run by comparator','inputs':{'historical_profiles':str(oldp),'historical_metrics':str(oldm),'new_profiles':str(ps[0]),'new_metrics':str(ms[0]),'hashes':{str(p):sha(p) for p in [oldp,oldm,ps[0],ms[0]]},'fullstate':fullstate},'tolerances':{'principal_relative_L2':1e-4,'principal_max_difference_over_shared_peak':5e-4,'x_and_mask':'exact','near_zero_collision_and_conservation_metrics':'all differences reported, no relative-identical claim'},'rows':len(old),'profile_columns':columns,'metric_columns':metrics,'historical_metrics_recomputed':internal_old,'new_metrics_recomputed':internal_new,'principal_profiles_within_declared_backend_tolerance':principal_pass,'x_and_mask_exact':structural_pass,'CSV_metric_self_consistency':internal_pass,'qualified_backend_comparison':principal_pass and structural_pass and internal_pass,'limitations':['A CPU regeneration does not establish stationarity or positivity of the signed Shakhov target.','Same fullstate is independently bound only when --fullstate hash is supplied; compare source SHA in provenance.','Frozen R/Delta use df=f-Md while S_flux uses continuum5rhoT^2; plateau[S_flux-S]=plateau[Bxx(Md)-5rhoT^2] apart reduction error. This reference correction cannot be inferred as a validated new solve.','Mean target heat ratio and conservative defects are compared as stored; CPU float32 backend can change tiny cancellation metrics.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['rows','principal_profiles_within_declared_backend_tolerance','x_and_mask_exact','CSV_metric_self_consistency','qualified_backend_comparison']},indent=2))
if __name__=='__main__':main()
