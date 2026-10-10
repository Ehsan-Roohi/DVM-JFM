# Full-distribution follow-up, 2026-10-10

## Two reproducibility questions

The original BGK fullstates were recovered and SHA-256 verified. Their archived heat-budget moment columns and scalar-budget inputs were regenerated from the distributions in bounded float64 chunks. The original Figure 5 fullstate was independently SHA-bound and processed with the unchanged archived Shakhov postprocessor and target builder on a single Unity CPU. No shock was evolved or declared stationary in this follow-up.

## BGK contracts and changed numbers

Stored spatial coordinate is x_mfp and computational relaxation time is tau=1. The scaled-coordinate kn_eff must not be combined with x_mfp. Heat derivatives use one binomial (1,4,6,4,1)/16 pass on u only. The actual scalar CSV uses a seven-point boxcar on J,u,K, not the unsmoothed script default. This setting was recovered numerically from its derivative columns and reproduced to roundoff. Each display subtracts the first/last8% endpoint mean. The comparison below freezes the original mask abs(plateau(S_original_rec))>=.02peak, so changing the active mask cannot cause the reported changes.

Corrected collision projections retain q(Md) and C4(Md), using the actual stored velocity coordinates/weights and a specified four-iteration double-precision conservative target. The heat flux is integrated from Fq_direct(x[0]), with its exact geometric projection A=sum(w vx(.5c²+cx²)f). The scalar reconstruction is Delta_rec=C4(Md)-15rhoT²-tau*(dJ/dx+4u'K). Consistently recomputed f-macros define the primary corrected numbers; the stored-centre variant is also deposited and gives almost identical results. This is an explicitly specified diagnostic target, not a claim of byte-identical historical float32/CUDA target history.

| Mach | Historical R error % | Corrected R error % | Historical Delta error % | Corrected Delta error % |
|---|---:|---:|---:|---:|
| 2 | 4.089892 | 4.426870 | 3.833580 | 3.757490 |
| 3 | 3.536827 | 3.734811 | 3.435461 | 3.456750 |
| 5 | 2.427902 | 2.543526 | 2.151008 | 2.182210 |

The historical values reproduce the pre-follow-up manuscript table after rounding. The current article uses the corrected numbers without a lengthy version-comparison discussion. They are not the exact-discrete corrected values. The anchor correction is a constant removed by plateau subtraction; q(Md) has a very small effect. The scalar Maxwellian fourth-moment defect is consequential. Independent receipts isolate this effect using the same mask and heat reconstruction. Corrected means, temperature and mass constraints, both masks, actual anchors, source magnitudes, and all input hashes are in each CONTRACT_RECONCILIATION.json. RESULT.json's unsmoothed scalar reconstruction is preserved only as a superseded diagnostic, never as a qualified table.

## Figure 5 lineage

Original fullstate SHA-256: 9077c010ac9169e573e8b47c55864496f220ea9f5c5f517b08ac90603fb45011 (95,780,214bytes). Frozen postprocessor SHA-256: 7a5c89202baf85b577812da799a7a9c485147ec3cf131ad82695858ee4e2bc3e; target builder: b16244133c3674304a31cca716c7829dcc846f05c36485bf94f5b08aee7913ff. Unity job65521579 completed0:0, one CPU, 8GiB allocation, monitored postprocess272.238s and minimum available memory4.778GiB. Live I/O used the workspace returned by ws_list, expires2026-11-06; project files were read-only inputs/dependencies.

All1800 rows and all17 numeric profile columns were checked. The spatial coordinate and active mask match exactly. Twelve principal physical/reconstruction columns agree within the prospectively declared CPU/CUDA tolerance (relativeL2<=1e-4 and max difference/shared peak<=5e-4); the largest principal relativeL2 difference is 1.6821232e-05. CSV norms reproduce their own metrics. Tiny cancellation quantities such as Delta_target are reported in full rather than falsely described as relatively identical. All downloaded output hashes are checked against the terminal scratch manifest. The rebuilt PDF/PNG reproduce the numerical curves, not the metadata bytes of the old PDF.

The frozen figure uses discrete-reference R,Delta from f-Md but a continuum5rhoT² subtraction in S_flux. Their remaining flux-channel discrepancy contains the discrete Maxwellian reference defect plateau[Bxx(Md)-5rhoT²], in addition to reduction differences. It should not be presented as a steady spatial-budget error. The signed Shakhov target minimum remains negative. Full-distribution regeneration closes the numerical lineage gap, not positivity or stationarity qualifications.

## Reproduction and deposit

Keep archived snapshots distinct from corrected postprocessing. Place the hash-matched BGK_M2.npz, BGK_M3.npz, BGK_M5.npz under fullstates/ next to these scripts. Run `python regenerate_bgk_budgets.py --mach M --chunk 4`, then `python reconcile_scalar_window.py --mach M` and `python review_reconciled_budget.py --mach M`, where M is 2, 3 or 5. These fullstate computations are locally RAM-gated; no kinetic time stepping occurs. Their compact reductions and reviewed reports are public. The larger BGK full distributions remain preserved locally and in the archive outside Git; their public input-data distribution remains an explicit deposit limitation. The full Figure 5 input is included publicly at research/raw_inputs/Figure5_M3_Shakhov.npz, with its SHA-256 recorded in the main publication manifest.

The frozen Figure5 source can be run directly on a compute node using the command in EXECUTION_RECEIPT.json after the input/source hashes are verified. Adapt paths to an allocated workspace; do not use /project for live I/O. compare_figure5.py checks the regenerated CSVs against the historical deposited ones. No credentials or keys are included.

These are reconstructions of finite-time numerical profiles. Neither a differential budget nor a visually regenerated plot certifies steady, spatial-grid, temporal-grid, or velocity-tail convergence.
