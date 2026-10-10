from pathlib import Path
import hashlib,json
F=Path(__file__).resolve().parent;D=F/'figure5_regenerated'
rows=[]
for line in (D/'OUTPUT_SHA256SUMS').read_text().splitlines():
    h,name=line.split(None,1);p=D/name.strip();actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==h,(name,h,actual);rows.append({'path':name.strip(),'bytes':p.stat().st_size,'sha256':h})
r=json.loads((F/'FIGURE5_CPU_REGEN_COMPARISON.json').read_text());e=json.loads((D/'EXECUTION_RECEIPT.json').read_text())
assert r['qualified_backend_comparison'] and r['rows']==1800 and len(r['profile_columns'])==17
assert e['exit_code']==0 and e['no_shock_evolution'] and e['minimum_available_GiB']>=1
assert r['inputs']['fullstate']['sha256']=='9077c010ac9169e573e8b47c55864496f220ea9f5c5f517b08ac90603fb45011'
principal=['rho','ux','T','qx','sigma_xx','R','Delta','S','S_from_flux','Delta_from_source','R_delta0','R_two_channel']
summary={'status':'PASS_INDEPENDENT_FIGURE5_CLOSURE_READBACK','hashes_verified':rows,'rows':r['rows'],'profile_columns':len(r['profile_columns']),'principal_max_relative_L2':max(r['profile_columns'][k]['relative_L2'] for k in principal),'principal_max_peak_scaled_difference':max(r['profile_columns'][k]['max_over_scale'] for k in principal),'x_and_mask_exact':r['x_and_mask_exact'],'metric_CSV_internal_consistency':r['CSV_metric_self_consistency'],'elapsed_s':e['elapsed_s'],'minRAMGiB':e['minimum_available_GiB'],'scientific_limits':'CPU float32 agrees within explicitly declared bands, not byte identity; original signedtarget negativity and finite-time state retained; no shock evolution/stationarity evidence.'}
(F/'INDEPENDENT_FIGURE5_CLOSURE_REVIEW.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
