# Fullstate heat/scalar budget closure

All three parent-authorized archival postprocessing cases completed with exit0 (129.17/260.67/461.19s). No shock evolution. Each used chunk4, one CPU/BLAS1, fresh nativeRAM>=4GiB and liveRAM>=1GiB gates, external600s limit.

Use BGK_M{2,3,5}_regenerated/CONTRACT_RECONCILIATION.json and scalar_window7_reconciliation.npz for final scientific numbers. Original RESULT.json scalar-window1 outputs are explicitly unqualified and superseded.

## Provenance and original reproduction

Actual downloaded original full distributions reproduce archived heat moments and heat reconstruction to floating-point roundoff. The scalar CSV was generated with boxcar7 on J,u,K, not script default1; this exact contract is independently inferred and reproduced. The initial window1 mismatch was diagnosed and retained rather than concealed.

| Mach | Heat max abs across stored columns | Scalar reconstructed Delta max abs | Legacy R two-budget error | Corrected same-mask R error | Corrected scalar error |
|---|---:|---:|---:|---:|---:|
| 2 | 5.68e-14 | 9.15e-13 | 4.089892% | 4.426942% | 3.757625% |
| 3 | 3.41e-13 | 3.64e-12 | 3.536827% | 3.734781% | 3.456756% |
| 5 | 5.33e-15 | 2.19e-11 | 2.427902% | 2.543511% | 2.182241% |

## Meaning of the corrections

Corrected heat source is q[Md]-q[f] with computational tau1 in x_mfp; anchor is actual Fq[0]; geometry is directly reduced at the specified centre. Corrected scalar source retains C4[Md], rather than assuming continuum15rhoT2. All archived plateau/mask contracts are explicit; both fixed legacy and new corrected-S masks are reported.

The source and scalar corrections do not remove finite-time/spatial-discretization errors. Their small effects do not establish stationary shocks. The large one-channel R error remains about63--64%, while two-channel recovery remains a few percent. This supports the channel-complement conclusion with revised quantitative diagnostics, not exact recovery or convergence.

Md is reconstructed in float64 with four bounded Newton iterations on the actual stored velocity grid/weights. Target mass/mean/temperature errors versus requested macros are recorded in each contract. Original float32/CUDA target identity is not asserted. Stored-centre and consistently recomputed-macro results remain separate; no silent recentering or normalization.

Fullstate/file hashes, archived-CSV reproduction, target checks, source/anchor sizes and correlations are in the machine-readable contracts. Distribution memmaps and originals are preserved locally; exclude huge extracted distribution.npy files from ordinary Git.
