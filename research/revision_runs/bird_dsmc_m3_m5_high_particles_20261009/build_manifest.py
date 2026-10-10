from pathlib import Path
import hashlib, json

R = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256((R / p).read_bytes()).hexdigest()
cases = []
for task, mach in enumerate((3, 4, 5)):
    cases.append(dict(task=task, Mach=mach, seed=202610092 + mach - 2,
                      nx=800, upstream_ppc=2000, dt=.005, L=15*mach,
                      warmup_time=200, final_time=400, sample_interval_steps=20,
                      block_steps=100, warmup_blocks=400, total_blocks=800))
names = ['DSMC1S_original.for','DSMC1S_paper.for','paper_diagnostics.f90',
         'postprocess.py','preflight.py','chain.py','plot.py','run.sbatch','gate.sbatch']
m = dict(campaign='bird_dsmc_m3_m5_high_particles_20261009',
         scratch='/scratch3/workspace/roohie_umass_edu-jfm1439-r26-m2-20261007/bird_dsmc_m3_m5_high_particles_20261009',
         archive='/project/pi_roohie_umass_edu/BGK_shock/revision_20260929_gpu_preempt/bird_dsmc_m3_m5_high_particles_20261009',
         purpose='Independent high-particle-count Bird DSMC Mach 3, 4, and 5 runs at 2000 upstream particles/cell; same method and sampling as the M2 high-particle case.',
         cases=cases,
         collision='Bird VHS, omega=1, inverse VSS alpha=1 (isotropic); unchanged from original campaign',
         transport='mu*=T*, tau*=1/rho*; same physical transport and boundary method',
         particle_capacity=5000000,
         particle_capacity_change_only='All 15 MNM declarations 500000 -> 5000000; replacing back yields byte-identical old DSMC1S_paper.for source',
         comparison_baseline='bird_dsmc_m2_m5_20261008: original 100-ppc runs at M3/M4/M5, same 800 cells, dt=.005, t=400',
         limits='One independent seed per Mach; finite-time and particle-count sensitivity, not confidence intervals or physical model validation',
         sources={n:sha(n) for n in names})
(R/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha('manifest.json'),'sources':m['sources']},indent=2))
