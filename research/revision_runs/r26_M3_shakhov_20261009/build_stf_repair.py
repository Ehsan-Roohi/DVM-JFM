"""Correct the three explicitly printed STF brackets, in an isolated copy."""
from pathlib import Path
import shutil,json,hashlib
HERE=Path(__file__).resolve().parent;SRC=HERE/'physics';OUT=HERE/'stf_repaired_tensor';OUT.mkdir(exist_ok=True)
shutil.copyfile(SRC/'shock_pilot.py',OUT/'shock_pilot.py');(OUT/'cavity_source').mkdir(exist_ok=True)
for f in (SRC/'cavity_source').glob('*.py'):shutil.copyfile(f,OUT/'cavity_source'/f.name)
p=OUT/'cavity_source/r26_tensor_closures.py';text=p.read_text(encoding='utf-8')
old='eq25_plus8_div_u = 8.0 * np.einsum("...abc,...->...abc", mm, div_u)'
new='''# Eq25 bracket extends over FOUR indices ijkm, before divergence.
    eq25_plus8_div_u = 8.0 * np.einsum("...dijkd->...ijk", stf4_project(np.einsum("...ijk,...dl->...dijkl", mm, gu)))'''
assert old in text;text=text.replace(old,new)
old='t2_raw = np.einsum("...b,...ba->...a", q, gu) + np.einsum("...b,...ab->...a", q, gu)'
new='''# Both Eq26 velocity gradients carry the two-index STF brackets.
    t2_raw = 2.0 * np.einsum("...b,...ab->...a", q, stf2_project(gu))'''
assert old in text;text=text.replace(old,new);p.write_text(text,encoding='utf-8')
p=OUT/'cavity_source/r26_bulk_equations.py';text=p.read_text(encoding='utf-8')
old='r_deformation = 6.0 / 7.0 * rr * div_u[..., None, None]'
new='''# Eq20 bracket extends over THREE indices ijk, before divergence.
    r_deformation = 6.0 / 7.0 * np.einsum("...dijd->...ij", stf3_project(np.einsum("...ij,...dk->...dijk", rr, gu)))'''
assert old in text;text=text.replace(old,new);p.write_text(text,encoding='utf-8')
mixed=(HERE.parents[1]/'revision_runs/r26_numerical_audit_20261008/mixed_prototype/implicit_mixed.py').read_text(encoding='utf-8')
mixed=mixed.replace("ROOT=Path(__file__).resolve().parents[2]/'r26_m2_pilot_20261007'/'numerical_repair_20261008'","ROOT=Path(__file__).resolve().parent")
(OUT/'implicit_repaired.py').write_text(mixed,encoding='utf-8')
hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*.py')}
(HERE/'stf_repair_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps(hashes,indent=2))
