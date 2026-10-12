from pathlib import Path
import hashlib,json
import bgk
P=Path(__file__).resolve().parent
bgk.gate()
import native_gate
files=['bgk.py','bgk_reference.py','shakhov_reference.py','kernel.c','kernel.so','native_kernel.py','native_gate.py','prepare_gate.py','manifest.json','resource_watchdog.py','PASS.json','NATIVE_GATE.json','run.sbatch']
(P/'IMMUTABLE_SHA256SUMS').write_text(''.join(hashlib.sha256((P/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in files))
print('Higher-Mach cold-map oracles qualified; no physical production trajectory has been accepted by these tests.')
