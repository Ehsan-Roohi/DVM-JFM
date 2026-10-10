# Critical solver-specification audit, v19 (2026-10-10)

## Scope

Read main.tex from abstract through all appendices, with numerical claims checked against the bundled M2/M3 raw profiles, M3 run/check/manifest receipts, original shock project's source files and recorded configurations. Edited only the isolated audit manuscript main.tex. No numerical evolution or downloaded code was executed. Original manuscript ZIP and original project were not modified. GitHub/data-availability wording remains the parent's responsibility.

## Correct collision-model / Mach mapping

Prandtl number is not selected by Mach number. Different collision models coexist at the same Mach.

| Campaign | Mach numbers actually used in this manuscript | Pr / transport |
|---|---|---|
| BGK two-budget sweep | 2, 3, 5 (no Mach-4 row) | Pr=1, tau=1, mu=p*tau |
| BGK sensitivity | 5 | Pr=1, tau=1; prescribed counts, not stationary stop |
| BGK high-Mach full/lightweight | 6, 8, 12 | Pr=1, tau=1; coarse transverse quadrature defects retained |
| Transport-matched Shakhov DVM | 2, 3, 4, 5 | Pr=2/3, omega=1, mu1=1, mu=T, tau=1/rho |
| Corrected Shakhov R26 curves | 2 and 3 only | Pr=2/3, mu=T, tau=1/rho; steady residual solve |
| Published-curve Shakhov benchmark | 8 | Pr=2/3, omega=.81, mu_ref=1.0338373584; g/h reduced representation |
| ES-BGK / ES-FP | source evaluations on BGK M2, M3, M5 | alpha_ES=-1/2 / beta=-5/4; no independent ES shock evolution |
| Original/corrected sampler DSMC source ensemble | 3 | Hard-sphere collision model, not an imposed Pr relaxation target |
| High-particle DSMC profiles | 2, 3, 4, 5 | VHS omega=1, alpha=1 isotropic scattering; one seed per Mach |

The previous assistant statement suggesting all Mach-2--5 calculations were Pr=2/3 was therefore too broad. The core budget reconstruction is BGK Pr=1; the separate cross-method profiles are Shakhov Pr=2/3. Both occur in the paper.

## Implemented corrections

1. Added an early solver-campaign table and explanatory paragraph, explicitly associating Pr, Mach, relaxation/viscosity law and evolution versus postprocessing scope.
2. Expanded the original ambiguous BGK transport paragraph to name its Mach campaigns and explicitly distinguish the M2--5 Shakhov campaign.
3. Abstract and conclusion now identify the tested budget-recovery cases as Mach 2, 3 and 5, rather than a continuous Mach-2--5 sweep. Conclusion explicitly excludes an unperformed Mach-4 budget test and the separate Pr=2/3 campaign.
4. Added actual cross-method Shakhov implementation details: g/h/k transverse integrations, uniform cell centres, composite-trapezoidal longitudinal quadrature, first-order upwind, analytic exponential collision split, CFL=.8, no clipping, RH incoming reservoirs, actual plotted grid extents and saved M3/M4/M5 times.
5. Added the different Shakhov cross-method stationarity criterion: eight fields, denominator floor 1e-8, checks roughly every two physical time units, five consecutive changes below 1e-7. This differs from the Mach-8 five-field/check-every-2000-step criterion.
6. Defined the meaning of R26 numerical acceptance from the M3 manifest/checker: p4 representation, 4012 elements, 16049 native points, q8 residual objective, sparse augmented LM, native corrected tensor PDE/BC qualification, measured PDE .0046433 and BC/phase 1.51e-11. Runner's iteration-limit/automatic-acceptance-false receipt is explicitly distinguished from the independently passing checker. No zero residual, uniqueness or collision-operator validation is claimed.
7. Added the channel-check limitation: source-fixed archived CSVs establish the 8-percent endpoint-mean subtraction and 2-percent active mask. A bounded CSV-only verifier reproduces all four reported norms, proves the raw/source display offset, and records exact input hashes. Full-distribution regeneration remains pending; negative target minima mean these channel diagnostics do not certify a stationary positive-target Shakhov solution.
8. Added explicit heat-budget reconstruction limitation: source script includes moving-frame term, but uses continuum source simplification and continuum upstream anchor; the new exact discrete formulas have not been rerun in the corrected guarded script. Retained quantitative errors are archived values, not fresh exact-discrete recomputations.

## Evidence paths

Paper project root (read-only):
`C:/Users/roohie/OneDrive - University of Massachusetts/Documents/ChatGPT/ریوایز جی اف ام tail`

- `revision_runs/source/dvm_bgk_normal_shock_conservative.py`: tensor-product trapezoid weights; first-order upwind; discrete Maxwellian matching; collision update exp(-dt), proving unit computational relaxation.
- `revision_runs/source/dvm_bgk_normal_shock_conservative_hmom_densemicro.py`: same unit-rate update and dx/CFL/vmax step formula.
- `revision_runs/source/dvm_revision.py`: Mach-5 sensitivity source, exp(-dt), optional transverse bound, explicit CFL.
- `revision_runs/dvm_r26_transport_match_20261007/manifest.json`: M2 base/refinements, Pr2/3, omega1, mu1=1, CFL=.8, target time400; x1280 plotted profile geometry.
- `revision_runs/dvm_r26_transport_match_20261007/solver.py`: g/h/k moments, tau=mu1*T^(omega-1)/rho, exp(-dt/tau) and exp(-Pr*dt/tau) terms, no clipping, incoming RH reservoirs, eight-field stationarity criterion.
- Audit package `manuscript/overleaf/review/M3_raw/DVM_receipt.json`: source hash `9a20d251...`, M3/M4/M5 configs and snapshot times238.125/173.4375/143.4375; zero stable count, stationary_certified=false.
- Audit package `review/M3_raw/R26_manifest.json`: Pr2/3, mu=T, tau=1/rho, p4/q8, native tensor criterion .01, BC criterion1e-4.
- Audit package `review/M3_raw/R26_RESULT.json`: ITERATION_LIMIT, automatic accepted=false, independent checker pending at runner close.
- Audit package `review/M3_raw/R26_independent_check.json`: subsequent independent_numerical_pass=true, native PDE .0046432647401, profile SHA `1b87004303b2d3a5e9c2fc377b2e7eca46e2151a24570b4119237337306b8768`.
- Audit package `review/M2_raw/R26_accepted_Shakhov_M2.npz`: 3841 native points; M2 curve identified by corrected-comparison JSON, but raw NPZ alone has no complete run manifest.
- `revision_runs/m8_reference/config.json`: Pr2/3, omega=.81, CFL=.7, float64, clipping=false,1200cells/401velocities, checks2000/threshold1e-7/five.
- `revision_runs/m8_reference/solver.py`: mu_ref formula/tau law, g/h collision exponential, independent homogeneous checks; matches existing appendix specification.
- `revision_runs/model_audit/source_snapshot/src/week2_jfm_upgrade/postprocess_shakhov_channel_consistency.py`: plateau_zero applied to R/Delta/S and recovered scalar; Pr2/3, Kn1 defaults.
- `revision_runs/model_audit/corrected_20261001/inputs/M3.metrics.csv`: matches retained Figure5 numerical values; min_fS=-9.9566837889e-5; source neutrality relative error2.072e-7.
- `revision_runs/model_audit/corrected_20261001/provenance.json` and `report_FA.txt`: corrected plotting input-column selection, source normalization guard NOT rerun on fullstate.
- `revision_runs/model_audit/corrected_20261001/exact_heatflux_budget_from_fullstate.py`: moving-frame term present, source-q/kn simplification and continuum edge-averaged anchor remain.
- `revision_runs/source/dsmc_revision.cpp`: constant total cross section and isotropic elastic scattering, confirming hard-sphere ensemble model; pair selection limitations already acknowledged.

## Critical remaining issues / publication gates

### High priority: retained budget numbers and exact discrete formulas

The manuscript now correctly states q[M_d] and discrete anchor terms, but the archived calculation simplifies them. Exact identity of every Figure2--4 input to the newly stated discrete formulas is not established, and the source normalization guard has not been rerun on fullstate. A repository containing scripts alone does not resolve this scientific gap. Reproduce the retained figures and errors from the full distributions with explicit coordinate/source scaling, moment-reference convention, moving-frame term, actual anchor and active masks before calling them fully audited exact-discrete errors. No silent change to reported numbers was made.

### Figure5: CSV convention resolved; full-distribution regeneration pending

The exact profile SHA256 is `c0fa17cc456bd40b5227fa710a36080fcc8473315f20e7c5002bd58e29971a91`; metrics SHA256 is `29b46b616e1ebbc2a6bde3ca42e22a335cac58ddf211f43378cbe197ff00c1f1`. These match the originally collected source-fixed evidence files `evidence/3.csv` and `evidence/2.csv`. `verify_figure5_csv.py` and `FIGURE5_CSV_VERIFICATION.json` record bounded, independently inspectable reproduction from these profiles: 1800 points, first/last144 used for the plateau means,680 active points, all six channel endpoint means below7e-18, and all four norms agreeing with archived metrics. The quantity Delta_from_source + Kn*Qdelta_direct is a constant0.0681607720468188 to3.6e-15, proving the reported reconstruction is plateau-corrected rather than raw. The relations S=R+Delta/3 and R_two=S_flux-Delta_source/3 also pass. No kinetic calculation or fullstate processing was done.

The retained Figure5 PDF hash `8ae23c83203d772b6fd03431077f3d573018b38927b78c7fe8664435584a5139` matches the Oct5 delivery manifest; the Oct1 corrected PDF has a different hash, so exact figure regeneration must not be inferred from the corrected plotting source alone. Publication should retain the exact CSVs and bind their original fullstate/generation command/source snapshot. The remaining limitation is upstream generation provenance, not uncertainty about the archived CSV plateau convention.

### Scope / convergence

- BGK high-Mach integrations stop at prescribed counts. Their full/lightweight agreement is combined sensitivity, not isolated convergence. Mach12 upstream stress defect .8413rhoT is severe; original manuscript already identifies it.
- Mach8 Shakhov stationarity is unmet and negative reduced-distribution fractions remain ~2.14% (g) and5.2% (h). Density/temperature agreement is bounded macroscopic comparison only. The g/h pair does not determine the transverse fourth moment k or all3D fourth-order channels.
- Cross-method DVM snapshots are not stationarity-certified. Label `finished` merely describes a prescribed end or threshold and is not equivalent to stationary unless stable count satisfies the criterion.
- M3 R26 independently passes predeclared finite tolerances, despite runner `accepted=false`; this distinction has now been made explicit. M2 detailed acceptance manifest/checker should be bundled for equally complete solver reporting.
- The provided manuscript supplies no corrected R26 M4/M5 curves. Additional successful runs in another conversation must not be attributed to these figures without source/receipt/profile identity and a manuscript update.
- ES-BGK/ES-FP source laws are evaluated on BGK fields, not independent shock solutions. Pr2/3 coefficients do not make the underlying BGK profile a Pr2/3 profile.
- DSMC high-particle VHS omega1 campaign differs from the hard-sphere production ensemble. Neither uses a tunable Shakhov Pr input; transport matching does not identify fourth-order production. Existing one-seed/sampler/stationarity cautions are preserved.

### GitHub/reproduction

Parent/second agent owns actual repository inventory and publication. Public repository must distinguish exact archived source snapshots from corrected analysis sources, bind figures to profile hashes, and include explicit run commands/configs. No assertion that all numerical material is public was introduced by this agent.

## Validation

Read-only source/receipt inspection and LaTeX patch review completed. Only archived CSV norms and algebraic identities were recomputed; no physical solution/fullstate was regenerated and no solver launched. The campaign table has five columns and four separators per data row; specified widths total0.91 linewidth plus30pt padding. Parent should compile the full project and visually inspect the added table after integrating the separate repository paragraph.
