# DVM-JFM

Reproducibility repository for *Closure-channel identifiability and two-channel recovery in monatomic kinetic normal shocks*.

This repository deposits the recoverable scientific sources, compact numerical profiles, tables, figure inputs and provenance for the manuscript. It is an archival research deposit, not a claim that every campaign is stationary or that every script is immediately portable. Read the audit reports before using a result.

## Collision models and campaigns

| Campaign | Mach numbers | Model / transport |
|---|---|---|
| Main two-budget reconstruction | 2, 3, 5 | BGK, Pr=1, constant tau=1, mu=p tau |
| BGK sensitivity / high Mach | 5 / 6, 8, 12 | BGK, Pr=1; prescribed integration counts |
| Transport-matched cross-method DVM | 2, 3, 4, 5 | Shakhov, Pr=2/3, mu=T, tau=1/rho |
| Corrected R26 profiles in manuscript | 2, 3 | Shakhov production law, Pr=2/3; finite residual qualification |
| Separate published-curve benchmark | 8 | Shakhov, Pr=2/3, omega=.81; reduced g/h representation |
| ES-BGK and ES-FP diagnostics | BGK fields at 2, 3, 5 | Collision-source postprocessing; not independently evolved ES shocks |
| DSMC | Campaign-specific | Hard-sphere source ensemble or VHS omega=1 high-particle profiles; no imposed Shakhov Pr parameter |

Prandtl number belongs to the collision-model/transport specification, not to Mach number alone. Different models occur at the same Mach.

## Contents

- `manuscript/`: audited LaTeX project, figures, tables and bundled M2/M3 comparison inputs.
- `research/`: original relative campaign layout, source snapshots, corrected analyses, compact profiles/configurations and receipts. Historical and corrected versions are kept distinct.
- `provenance/publication_manifest.json`: SHA-256 and byte size of each deposited research artifact.
- `provenance/source_inventory.json`: source inventory, figure/table references and explicit raw-data exclusions.
- `audit/`: critical solver-specification and code/results coverage reports, plus validation records.
- `research/bird_dsmc_modifications/`: authored changes and external-original dependency documentation, where applicable.

## Recompute the bundled cross-method tables

With Python and NumPy installed, from the repository root:

```sh
python manuscript/review/recompute_dsmc2000_m2.py
python manuscript/review/recompute_dsmc2000_m3.py
```

These small postprocessing commands were rerun during the audit. Both resulting CSVs exactly matched the submitted CSV SHA-256 values. They do not certify DSMC stationarity, DVM stationarity or model error. Build the full LaTeX project from `manuscript/main.tex` with its bundled class, bibliography and figures; the audit machine had no TeX installation, so this revision has not been compiled there.

## Limits of this deposit

Multi-gigabyte full-distribution/restart files are not stored in Git. Their available paths and sizes are recorded in the inventory; a separate durable data deposit is still needed for complete restart reproduction. The compact profiles are results, not replacements for full velocity distributions in fourth-order source regeneration. Some archived drivers retain machine-specific paths and require explicit path adaptation. Third-party source or article assets are not given a new redistribution license by this repository.

Two scientific provenance gaps must be resolved before calling all manuscript results fully reproduced: the legacy budget numbers have not been regenerated using every newly stated exact discrete convention, and the source-fixed Figure 5 input needs regeneration from its original full distribution. The Figure 5 CSV arithmetic and 8%-endpoint plateau-subtraction convention were independently reproduced in the audit; this does not complete its fullstate-to-figure lineage. The audit reports explain these limitations. Do not infer corrected R26 Mach-4/5 results from this manuscript's Mach-2/3 curves.

No global convergence, uniqueness, nonexistence or validated high-Mach accuracy is implied by depositing an archived result. Cite a specific commit, not only a moving branch URL.
