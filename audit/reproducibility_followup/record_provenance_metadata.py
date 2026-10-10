from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
files=list((root/'frozen_figure5_source').glob('*'))+list(root.glob('M*_delta_budget.profiles.csv'))
rows=[{'local':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.is_file()]
out={'audit':'read-only source/log/archive provenance; no solver execution','files':rows,'figure5_candidate':{'archive':'/project/pi_roohie_umass_edu/BGK_shock/ref/shakhov_m3_cons/M3_shakhov_cons_RH_Pr23_fullstate.npz','bytes':95780214,'sha256':'9077c010ac9169e573e8b47c55864496f220ea9f5c5f517b08ac90603fb45011','duplicate':'M3_shakhov_cons_RH_Pr23_seg5.npz','exact_fullstate_to_CSV_binding':'pending_parent_numerical_regeneration'},'figure5_outputs':{'metrics':'29b46b616e1ebbc2a6bde3ca42e22a335cac58ddf211f43378cbe197ff00c1f1','profiles':'c0fa17cc456bd40b5227fa710a36080fcc8473315f20e7c5002bd58e29971a91'}}
review=root/'INDEPENDENT_FIGURE5_CLOSURE_REVIEW.json'
if review.exists():
    verified=json.loads(review.read_text())
    if verified['status']=='PASS_INDEPENDENT_FIGURE5_CLOSURE_READBACK':
        out['figure5_candidate']['exact_fullstate_to_CSV_binding']='CPU_frozen_source_reproduced_profiles_within_declared_float32_backend_tolerance_not_byteidentity'
        out['closure_review']=verified
(root/'Figure5_PROVENANCE_METADATA.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,indent=2))
