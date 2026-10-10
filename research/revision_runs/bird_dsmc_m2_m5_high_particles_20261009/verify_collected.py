"""Independently verify locally copied high-particle DSMC final records."""
from pathlib import Path
import hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parent
for mach in range(2, 6):
    d = ROOT / 'collected' / f'M{mach}'
    receipt = json.loads((d / 'RECEIPT.json').read_text())
    for name in ['profiles.npz', 'profiles.csv', 'analysis.json', 'status.json',
                 'progress.txt', 'case.in', f'chunk_{json.loads((d / "status.json").read_text())["job"]}.log',
                 'PASS.json', 'manifest.json']:
        assert name in receipt, (mach, name, 'not recorded in remote archive receipt')
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        assert got == receipt[name], (mach, name, got, receipt[name])
    status = json.loads((d / 'status.json').read_text())
    manifest = json.loads((d / 'manifest.json').read_text())
    passed = json.loads((d / 'PASS.json').read_text())
    assert status['finished'] is True and status['block'] == 800
    assert status['time'] >= 399.99
    assert status['manifest'] == hashlib.sha256((d / 'manifest.json').read_bytes()).hexdigest()
    assert manifest['cases'][0]['Mach'] == mach if mach == 2 else mach in [c['Mach'] for c in manifest['cases']]
    case = (d / 'case.in').read_text().split()
    assert int(case[0]) == mach and int(case[2]) == 2000 and int(case[5]) == 800
    z = np.load(d / 'profiles.npz')
    assert len(z['x']) == 800
    for key in ['rho', 'u', 'theta', 'sigma', 'q', 'm', 'R', 'Delta', 'S']:
        assert np.isfinite(z[key]).all(), (mach, key)
    assert np.min(z['rho']) > 0 and np.min(z['theta']) > 0
    assert status['checkpoint'] == receipt['DSMC1S.RES']
    print(f'M{mach}: PASS files={len(receipt)} blocks=800 ppc=2000 '
          f'profile_sha256={receipt["profiles.npz"]} stationarity_certified={status["analysis"]["stationarity_certified"]}')

for mach in range(2, 6):
    d = ROOT / 'baseline_100ppc' / f'M{mach}'
    receipt = json.loads((d / 'RECEIPT.json').read_text())
    for name in ['profiles.npz', 'profiles.csv', 'analysis.json', 'status.json',
                 'case.in', 'PASS.json', 'manifest.json']:
        assert name in receipt, (mach, name, 'not recorded in baseline receipt')
        got = hashlib.sha256((d / name).read_bytes()).hexdigest()
        assert got == receipt[name], (mach, name, got, receipt[name])
    status = json.loads((d / 'status.json').read_text())
    case = (d / 'case.in').read_text().split()
    assert status['finished'] is True and status['block'] == 800
    assert int(case[0]) == mach and int(case[2]) == 100 and int(case[5]) == 800
    z = np.load(d / 'profiles.npz')
    assert len(z['x']) == 800 and np.isfinite(z['rho']).all() and np.isfinite(z['theta']).all()
    assert np.min(z['rho']) > 0 and np.min(z['theta']) > 0
    print(f'M{mach} 100-ppc baseline: PASS profile_sha256={receipt["profiles.npz"]}')
