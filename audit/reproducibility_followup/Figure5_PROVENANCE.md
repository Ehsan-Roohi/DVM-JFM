# Figure5 full-distribution genealogy

Follow-up, 2026-10-10. No solver was advanced. Exact archived source/logs were transferred to `frozen_figure5_source/`; parent completed the bounded frozen-source postprocessing and this agent independently verified its local readback, hashes and numerical comparison.

## Exact output evidence

Archive directory: `/project/pi_roohie_umass_edu/BGK_shock/figures/week2_jfm_upgrade_shakhov_cons/M3_shakhov_cons_channel_consistency_sourcefixed_latest/`.

* `.profiles.csv`:486,219bytes, SHA256 `c0fa17cc456bd40b5227fa710a36080fcc8473315f20e7c5002bd58e29971a91`.
* `.metrics.csv`:740bytes, SHA256 `29b46b616e1ebbc2a6bde3ca42e22a335cac58ddf211f43378cbe197ff00c1f1`.
* Exact remote hashes match locally collected `model_audit/evidence/3.csv` and `2.csv`. `evidence_index.json` records that mapping.
* Archived sourcefixed log reports the same four errors, conservation defects, minimum distribution and signed-target minimum. Output timestamp2026-05-29 21:06UTC.

## Candidate input recovered

`/project/pi_roohie_umass_edu/BGK_shock/ref/shakhov_m3_cons/M3_shakhov_cons_RH_Pr23_fullstate.npz`

95,780,214bytes; SHA256 `9077c010ac9169e573e8b47c55864496f220ea9f5c5f517b08ac90603fb45011`; mtime2026-05-29 20:32:03UTC. `M3_shakhov_cons_RH_Pr23_seg5.npz` is byte-identical, mtime20:29:08UTC. Parent downloaded and independently hash-checked this input.

ZIP directory lists16arrays, total461,955,856uncompressedbytes: `x`, `rho`, `ux`, `T`, `qx`, `sigma_xx`, `Rxx_closure`, `Delta`, `S_R_plus_Delta3`, `f`, `v`, `w`, `Pr`, `Kn`, `left_state`, `right_state`. `f.npy` alone460,864,928bytes. The segment5 log gives `nx=1800,nv=64009,dx=.06670372188091278,Pr=2/3,Kn=1`. This is a full3V distribution, not a reduced profile NPZ.

Segment5 started from `..._seg4.npz`, used500sweeps, `omega=.04`, target chunk32, CUDA, exact monatomic Mach3 RH states: left`(rho,u,T)=(1,3.872983346207417,1)`, right`(3,1.2909944487358056,3.6666666666666665)`. Its terminal target defects and negative target minimum match sourcefixed metrics. The companion launcher copies the latest segment to `_fullstate.npz`.

The historical sourcefixed log does **not** record its `--fullstate` argument or full command. Its initial genealogy was therefore a strongly supported candidate, pending regeneration. That regeneration has now completed successfully: Unity job65521579 used the hash-bound9077input and exact frozen modules on CPU, with no shock evolution. All1800rows/all17profilecolumns were compared; x and active mask match exactly. Principal profile relative-L2 differences are at most1.6821232182e-5 and shared-peak-scaled maximum differences at most1.3593699825e-5, within prospectively declared1e-4/5e-4backend bands. All stored-vs-recomputed CSV error metrics are internally consistent. This establishes a numerically matched input-to-profile binding across the float32CPU/CUDA backends, not original byte identity or recovered historical shell arguments.

`INDEPENDENT_FIGURE5_CLOSURE_REVIEW.json` verifies all7downloaded files against the remote `OUTPUT_SHA256SUMS`. The postprocess execution receipt reports exit0,272.2381s, minimum availableRAM4.77782GiB, one CPU, and no shock evolution. Historical signed-target negativity, finite-time state and the discreteMd/continuum-reference distinction remain; no stationarity or convergence claim follows.

Parent staged this Figure5 raw input for public publication at `research/raw_inputs/Figure5_M3_Shakhov.npz`, with SHA2569077c010... and95,780,214bytes. This is a specific exception to the larger raw-distribution exclusions: the three224.7/461.9/906.3MB BGK inputs still require a separate public data deposit. Repository push verification remains the parent's responsibility.

## Exact frozen code and reconstructed command

Remote sources at `.../BGK_shock/src/week2_jfm_upgrade/` were downloaded with exact SHA checks:

* `postprocess_shakhov_channel_consistency.py`: `7a5c89202baf85b577812da799a7a9c485147ec3cf131ad82695858ee4e2bc3e`.
* `run_shakhov_conservative_sweep.py` (imported moment/target functions): `b16244133c3674304a31cca716c7829dcc846f05c36485bf94f5b08aee7913ff`.

The postprocessor casts input/grid to float32 and reconstructs the same signed conservative discrete target. Its collision scalar is raw `sum(w*c^4*(fS-f))/Kn`, followed by plateau-zero subtraction; it does not obtain the scalar source by differentiating a flux. The target constructs a discrete Maxwellian, projects the Shakhov correction against mass/momentum/energy, and scales its heat flux to`(1-Pr)q`. Transport iteration floors `f` at1e-30 by default, while the target itself is signed. Thus tiny target scalar-excess agreement is a discrete diagnostic; it does not imply a positive target or a stationary solution.

Reconstructed postprocess command (in an isolated scratch directory with both frozen modules under `src/week2_jfm_upgrade`; inferred from parser defaults and output case, not recovered shell history):

```bash
python src/week2_jfm_upgrade/postprocess_shakhov_channel_consistency.py \
  --fullstate INPUT_9077c010.npz \
  --case M3_shakhov_cons_channel_consistency_sourcefixed_latest \
  --Pr 0.6666666666666666 --Kn 1 --device cpu \
  --target-chunk 32 --moment-chunk 32 --active-thresh 0.02 \
  --outdir ISOLATED_OUTPUT_DIRECTORY
```

CPU backend/chunking can change float32 reduction order relative to the archived CUDA result. Record these changes; compare physical arrays/metrics with justified tolerances rather than demanding invented byte identity.

## Existing resource metadata, no launch

Actual `ws_list`: `/scratch3/workspace/roohie_umass_edu-jfm1439-r26-m2-20261007`, expiry2026-11-06 16:07:32; filesystem capacity read37,225,726,214,144freebytes. `cpu` and `cpu-preempt` available,14daylimits. Parent may choose a bounded1CPU8GB scratch-only job.

Historical interpreter `/work/.../.conda/envs/dde-tf/bin/python` is absent. Existing migrated interpreter is `/project/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dde-tf/bin/python`, Python3.11symlink; package file metadata reports PyTorch2.4.1, CUDA12.1. No torch import or scientific computation was performed on login during this lookup. Use the environment as read-only dependency; all live outputs go to the discovered scratch workspace.

## Related BGK budget inputs supplied to parent

Original archive prefix `/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep/`:

| File | Bytes | SHA256 |
|---|---:|---|
| `standing_M2_hmom_x40_nx1600_v97_19_19_vmax12_fullstate.npz` |224723455|`8959d23bfe7643d0010bedd65516c6985103b50e3f15c1cc862893180c770a02`|
| `standing_M3_hmom_x60_nx1800_v121_23_23_vmax16_fullstate.npz` |461949727|`b564ef786a7a5d4c6a7643ebdcc8a6dc053a1cacbd8fe66a8deeb1a1173bdf2b`|
| `standing_M5_hmom_x80_nx2200_v141_27_27_vmax22_fullstate.npz` |906261407|`fb234484d825fb24f3f97e2b774cd72b1ba185f7c5e0978e1f1e02ea8f09346d`|

The exact original `figures/week2_jfm_upgrade_delta_budget/M{2,3,5}_delta_budget/M{2,3,5}_delta_budget.profiles.csv` files were transferred to this follow-up folder (446865/495986/598417bytes). Their reproduction and heat-budget consistency are parent responsibilities.
