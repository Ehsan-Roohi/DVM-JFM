from pathlib import Path
import hashlib, json, subprocess
import numpy as np
import postprocess as pp

R = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
m = json.loads((R / 'manifest.json').read_text())
for name, expected in m['sources'].items():
    assert sha(R / name) == expected, name
results = {'moment_tests': pp.selftest(), 'cases': []}

def run(d, resume=False):
    with (d / ('resume.log' if resume else 'initial.log')).open('w') as log:
        subprocess.run([str(R / 'bird_shock')], cwd=d,
                       input='0\n0\n' if resume else '1\n1\n', text=True,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)

for c in m['cases']:
    mach = c['Mach']
    d = R / 'gate_runs_v5' / f'M{mach}'
    assert not d.exists(), f'gate directory already exists: {d}'
    (d / 'blocks').mkdir(parents=True)
    (d / 'case.in').write_text(f"{mach} {c['seed']} 10 .005 0 2 2\n")
    run(d)
    h, a = pp.load_block(d / 'blocks/block_000002.dat')
    f = pp.fields(a[:, 1:57], a[:, 57:92])
    assert np.isfinite(a).all() and np.min(f['rho']) > 0 and np.min(f['theta']) > 0
    q = a[:, 57:92]
    conserved = max(np.max(abs(q[:, 0])),
                    *(np.max(abs(q[:, pp.IDX[t]])) for t in [(1,0,0),(0,1,0),(0,0,1)]),
                    np.max(abs(q[:, pp.IDX[2,0,0]]+q[:, pp.IDX[0,2,0]]+q[:, pp.IDX[0,0,2]])))
    assert conserved < 1e-8, (mach, conserved)
    results['cases'].append({'Mach': mach, 'low_ppc_smoke': True,
                              'collision_conservation_max': float(conserved)})

    # Exercise the actual 2000-ppc particle capacity and restart path for each Mach.
    d = R / 'gate_runs_v5' / f'M{mach}_high_ppc'
    assert not d.exists(), f'gate directory already exists: {d}'
    (d / 'blocks').mkdir(parents=True)
    case = f"{mach} {c['seed']} {c['upstream_ppc']} {c['dt']} 0 2 2\n"
    (d / 'case.in').write_text(case)
    run(d)
    h, a = pp.load_block(d / 'blocks/block_000002.dat')
    f = pp.fields(a[:, 1:57], a[:, 57:92])
    assert np.isfinite(a).all() and np.min(f['rho']) > 0 and np.min(f['theta']) > 0
    high_sha = sha(d / 'blocks/block_000002.dat')
    e = R / 'gate_runs_v5' / f'M{mach}_restart'
    assert not e.exists(), f'gate directory already exists: {e}'
    (e / 'blocks').mkdir(parents=True)
    (e / 'case.in').write_text(f"{mach} {c['seed']} {c['upstream_ppc']} {c['dt']} 0 2 1\n")
    run(e)
    assert not (e / 'FINAL_TIME_REACHED').exists()
    run(e, resume=True)
    restart_sha = sha(e / 'blocks/block_000002.dat')
    assert restart_sha == high_sha, (mach, high_sha, restart_sha)
    results['cases'][-1].update(high_ppc=2000, high_ppc_finite_positive=True,
                                restart_block_bitwise=True,
                                high_ppc_checkpoint_sha=sha(d / 'DSMC1S.RES'))

results.update(PASS=True, manifest=sha(R / 'manifest.json'), sources=m['sources'],
               executable=sha(R / 'bird_shock'))
(R / 'PASS.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results), flush=True)
