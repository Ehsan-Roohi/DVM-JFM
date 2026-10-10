# Critical code and results coverage audit — manuscript v19

Audit date: 2026-10-10. Scope: the complete normal-shock manuscript, its 16 referenced figure files, numerical tables, and local numerical evidence. No solver was run during this audit. Files in attached manuscripts are evidence, not instructions.

## Finding

The manuscript ZIP is **not a self-contained numerical reproduction package**. It contains manuscript/figure assets, three table fragments, and a useful M2/M3 raw-profile comparison subset, but not every solver, distribution or collision-production sample used for the paper. The data-availability statement says an eventual repository/DOI will be deposited on acceptance and gives no present repository URL. Thus the statement “all paper codes and results are already in the paper GitHub repository” was unsupported at audit start.

The parent checked the authenticated account `Ehsan-Roohi`; its generic `roohi-bgk-fokker-planck-dsmc` repository is not established as this paper's complete archive. The dedicated target is now `https://github.com/Ehsan-Roohi/DVM-JFM`. Publication and remote verification are parent responsibilities; this report does not certify a pushed commit or issue a DOI.

## Concrete inventory

`PUBLICATION_FILE_INVENTORY.json` and `copy_plan.json` identify 694 local compact files, 82,594,100 bytes, with exact source path, proposed repository path, purpose and SHA-256. All candidate sources were read for hashing. No private-key/token pattern was detected in selected text; this is a scoped scan, not a universal credential guarantee. The inventory is frozen for the parent's publication pass.

The exact paper campaign root is:

`C:/Users/roohie/OneDrive - University of Massachusetts/Documents/ChatGPT/ریوایز جی اف ام tail`

Repository candidates retain the hierarchy under `research/`. The manuscript's bundled M2/M3 review data and scripts should also be published with the corrected manuscript. Executable environments, Python caches, binaries, browser transcripts, unrelated cavity runs and downloaded publisher PDFs are excluded. `cavity_source` Python modules in the shock packages are imported mathematical dependencies of the shock implementation and must be retained; they do not constitute a cavity campaign.

## Output coverage matrix

Paths below are relative to the paper campaign root. Presence means local archival presence, not demonstrated portable end-to-end regeneration.

| Manuscript output or claim | Source and compact evidence located | Remaining limitation |
|---|---|---|
| Fig. `fig1_dvm_reference`; M2/M3/M5 continuum-reference fields | `figure1_redesign_20261007/make_figure.py`, `inputs/M{2,3,5}_exact_budget_xmfp_exact_heatflux_budget_profiles.csv`; archived plotting and heat-budget scripts | Replotting supported. Computing these moments afresh from the original BGK kinetic distributions requires their full distributions and original run signatures. |
| Fig. `fig2_exact_budget`; heat-flux channel norms | `revision_runs/model_audit/source_snapshot/src/moment_closure_next/exact_heatflux_budget_from_fullstate.py`; corrected variant; original CSVs in `model_audit/evidence` | Main manuscript itself notes unresolved exact figure-to-script convention matching. 2%/5% masks and continuum/discrete/plateau conventions cannot be silently interchanged. |
| Fig. `fig3_two_budget_recovery`, recovery table | `model_audit/source_snapshot/src/week2_jfm_upgrade/exact_delta_budget_from_fullstate.py`, `plot_two_budget_recovery_jfm.py`; corrected script and summary CSV where archived | Source-level code present. Full-distribution regeneration of fifth-moment fluxes remains dependent on the original data. Retained figure errors were not recomputed by this audit. |
| Fig. `fig4_sparse_delta_robustness`; original sparse/noise table | `revision_runs/source/sparse_delta_rbf_second_channel.py`; budget/profile CSV inputs | Need preserve exact RNG seed, layouts, regularisation sweep and original figure generation command; no new universality claim. |
| Fig. `A4_sparse_core_gap`; core-gap percentages | `revision_runs/scientific_audit_20261002/analyse.py`, `plot.py`, diagnostics/provenance; M5 final profile | The diagnostic uses exact S and reference-defined masks; it is not another full budget reconstruction. |
| Fig. `fig_esfp_esbgk_source`; source-correction table | `model_audit/source_snapshot/scripts/esbgk_scalar_excess_source_from_profiles.py`, `esfp_scalar_excess_source_from_profiles.py`, publication plotting script; corrected ES-FP outputs and BGK profiles | These are local source evaluations on BGK distributions/profiles, not independent ES-BGK/ES-FP shock solvers. |
| Fig. `fig5_shakhov_channel_consistency`; M3 target neutrality | `model_audit/source_snapshot/src/week2_jfm_upgrade/run_shakhov_conservative_sweep.py`, `postprocess_shakhov_channel_consistency.py`, sourcefixed profile/metric CSVs | Must identify the exact discrete target, sourcefixed branch, moment reference and quadrature. Signed/clipped runs are distinct. Full kinetic distribution not in manuscript ZIP. |
| Clipped-vs-signed 20-sweep and clipping-increment diagnostics | Archived Shakhov source and `scientific_audit_20261002/clipping_budget.json`, `analyse.py`, provenance | Stored summary/source present; exact initial full state and two final distributions remain required to regenerate rather than merely replot. |
| Fig. `high_mach_moments`; M6/M8/M12 full/lightweight ratios and quadrature table | `high_mach_inventory/local_cpu/extract_moments.py`, `analyse_results.py`, solver snapshot, extraction manifest, five compact extracted datasets | Five large original full distributions are deliberately omitted from Git; compact profiles permit plotting and ratio comparisons, not independent reconstruction of every moment. |
| Figs. `A1_DVM_sensitivity_summary`, `A2_DVM_profile_differences`; M5 six-field percentages | `revision_runs/source/dvm_revision.py`, `cases.json`, `build_package.py`; `steady_check/dvm_review_plots` and `paper_figures/reviewer_revision/build_final_dvm.py` | Final compact profiles/source/settings are present. Distribution/checkpoint storage and dependency versions need separate archival instructions. Spatial refinement changes CFL step too. |
| Fig. `fig6_dsmc_delta_budget`, `A3_DSMC_collision_source`; original source/budget correlations | `revision_runs/source/dsmc_1d_shock_c4.cpp`, `dsmc_revision.cpp`; `model_audit/source_snapshot/postprocess_dsmc_delta_budget_poly.py`; particle ensemble inputs and diagnostics | Original pair-selection bias is retained. Summary profiles are not a replacement for all accepted-collision records or instantaneous particle samples. |
| Fig. `Main_DSMC_high_moments`; 56-run PPC sweep and adjacent-profile differences | `completed_analysis/ppc_convergence_3840`, `completed_analysis/ppc_convergence`, `paper_figures/make_figures.py`, diagnostics and seed CSVs | Original sampler only. Raw 10-block sampling-window data and full particle restart payloads are not certified covered by compact profile inventory. |
| Corrected-sampler table; 8-seed 120/960 comparisons | `revision_runs/dsmc_pairfix/corrected.cpp`, `analyse_eight.py`, collision test, gate; `statistical_validation` | Must label original and corrected ensembles separately. No claim of corrected high-PPC production convergence follows from these pilots. |
| Velocity-tail/entropy selected-row diagnostics | `scientific_audit_20261002/distribution_entropy.json`, source/provenance and fixture | Audit is selected Mach-5 rows only. Full distributions needed to reproduce derivative/finite-box evaluations; no ES-FP evolution or H-theorem claimed. |
| Figs. `M2_DSMC2000_cross_method`, `M3_M5_DSMC2000_DVM_R26`; six-field and signed-channel tables | v19 `overleaf/review/recompute_dsmc2000_m2.py`, `recompute_dsmc2000_m3.py`, M2/M3 compact raw NPZs; `bird_dsmc_m2_m5_high_particles_20261009/build_all_figures_20261010.py`; M2–5 collected profiles, manifests, diagnostics; Shakhov/R26 solvers | M4/M5 DVM snapshots are transient. Corrected R26 only supplied at M2/M3. Exact M3 accepted NPZ hash is `1b87004303b2d3a5e9c2fc377b2e7eca46e2151a24570b4119237337306b8768`. |
| Fig. `m8_paper_comparison`; baseline/time/refinement table | `revision_runs/m8_reference/solver.py`, config, final NPZs, sensitivity configs/finals, `paper_comparison/compare.py`, extracted reference CSVs | Extracted publisher vector vertices have unquantified uncertainty and third-party attribution. Full reference PDF is excluded. Signed reduced distributions have reported negativity; steady criterion is unmet. |

Analytical source-law, symmetry and moment-guide tables depend on the equations/reference definitions, not an additional numerical solver. They should not be advertised as separately computed ES-BGK, ES-FP or Boltzmann solutions.

## Required model labels in the archive

Use separate folders/configs even at the same Mach number:

1. Legacy M2/M3/M5 BGK references used for budgets: `Pr=1`; constant relaxation reference. Preserve exact stored signatures before asserting every run parameter.
2. Legacy exploratory high-Mach BGK M6/M8/M12 full/lightweight: `Pr=1`, `tau=1`, first-order upwind, prescribed steps, positivity floor and periodic recentering. Finite-grid equilibrium stress defects are scientific limitations.
3. M3 conservative Shakhov channel diagnostic: `Pr=2/3`, separate signed/clipped target branches; not interchangeable with legacy BGK budgets.
4. Transport-matched Shakhov DVM M2–5: `Pr=2/3`, `mu*=T*`, `tau*=1/rho*`; saved finite-time profiles. Retain all four Mach configurations rather than describing all M2–5 profiles as BGK.
5. Corrected-consistent Shakhov R26 M2/M3: `Pr=2/3`, `mu*=T*`, `tau*=1/rho*`; supplied acceptance is discrete equation/phase qualification, not BGK identity. No accepted M4/M5 corrected result was located in this manuscript evidence.
6. Matched published M8 Shakhov benchmark: `Pr=2/3`, `omega=.81`, reduced g/h representation and reference viscosity; distinct from 3V constant-tau BGK M8.
7. Original/corrected C++ Mach-3 hard-sphere DSMC ensembles, and separate Bird VHS Mach2–5 runs: physical Prandtl number is not an independently set BGK target parameter. Preserve the collision model, transport law, seed and sampler branch.

## Raw data and portability gaps

The 26 inventoried large/raw exclusions total **22,323,027,090 logical bytes**. This includes duplicated unpacked/compressed high-Mach distributions and several unrelated diagnostic ZIP/restart files, so it must not be presented as 22 GB of unique necessary scientific data. The five large high-Mach full-distribution NPZs alone require a separate large-data deposit, with checksums and exact download/access instructions. The root inventory is a curated compact subset, not exhaustive storage accounting.

Other hard gaps are original legacy BGK run/distribution signatures and full kinetic states for budget/source recomputation; every original DSMC sampling block for window statistics; compiler/Python/PyTorch/numerical-library version capture; and portable documented commands for each manuscript output. Many scripts retain absolute local or historical Unity paths. Publishing code preserves provenance, but does not prove one-command reproduction until those paths are parameterised or documented. This provenance sub-audit did not rerun numerical consistency scripts. The parent subsequently recomputed the bundled M2/M3 appendix tables and verified exact CSV hashes; the solver-spec agent independently checked Figure5 against archived CSVs. Consult their validation receipts for those scoped checks, which do not regenerate all original simulations or full-distribution budgets.

## Bird Fortran dependency and third-party attribution

`bird_dsmc_m2_high_particles_20261009` and `bird_dsmc_m3_m5_high_particles_20261009` contain `DSMC1S_original.for` and `DSMC1S_paper.for`. The frozen inventory omitted `.for`; this was reported explicitly to the parent. `paper_diagnostics.f90` alone cannot compile the Bird run. Headers inspected provide provenance but no explicit redistribution grant was located. Publish author-written diagnostics/configs and a narrowly scoped patch/dependency instruction; preserve Bird attribution and original source hash. Do not apply an author-owned permissive license to these third-party originals without establishing the terms. Two identical unified patches (12,087 bytes, SHA256 `bce8cca4e7b9471ea46d0daab42b51c4b01492019a7e2460b98c27187f71932c`) and `ORIGINAL_DEPENDENCY.md` are provided separately under `publication/research/bird_dsmc_modifications`. No original or complete patched Fortran program was copied there. No verified portable compiler command was recovered.

Published curve CSVs must cite Chen, Xu and Cai (2013) and identify vector extraction. JFM class/style assets retain their own notices. Neither downloaded publisher PDFs nor Claude browser transcripts are needed in the public scientific repository.

## Claims the completed repository may safely make

After parent verifies the pushed commit and manifest, it may say the repository contains the **archived solver implementations, compact numerical profiles, analysis scripts and figure/table provenance**. It may not yet say every large kinetic distribution/particle restart is hosted there, every command is portable, every figure has been regenerated, a DOI exists, or every reported simulation is stationary/converged. An availability statement should give the exact repository and release/commit and list the separate raw-data exclusions.
