from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
source=(HERE/'stf_repaired_tensor/cavity_source/r26_tensor_closures.py').read_text(encoding='utf-8')
fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='closures_from_tensors')
body=ast.get_source_segment(source,fn)
body=body.replace('coefficients = closure_coefficients(coefficient_mode)',"assert coefficient_mode == 'shakhov-derived'\n    coefficients = SimpleNamespace(C1=1.,C2=0.,Y1=1.,Y2=0.,Y3=0.,mode='shakhov-derived',provenance='Shakhov + STF-corrected Gu kinetic transport')")
body=body.replace('        + t8\n','        # Shakhov has no quadratic collision coupling.\n')
old=(HERE/'physics/r26_shakhov.py').read_text(encoding='utf-8');tree=ast.parse(old)
oldfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='closures_from_tensors')
oldbody=ast.get_source_segment(old,oldfn)
(HERE/'stf_repaired_tensor/r26_shakhov_tensor.py').write_text(old.replace(oldbody,body),encoding='utf-8')
print('Created independent STF-corrected 3D Shakhov kernel')
