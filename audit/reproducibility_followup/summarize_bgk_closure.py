"""Collect already completed contracts; no numerical evolution/reduction."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
summary=[]
lines=['# Fullstate heat/scalar budget closure','',
 'All three parent-authorized archival postprocessing cases completed with exit0 (129.17/260.67/461.19s). No shock evolution. Each used chunk4, one CPU/BLAS1, fresh nativeRAM>=4GiB and liveRAM>=1GiB gates, external600s limit.','',
 'Use BGK_M{2,3,5}_regenerated/CONTRACT_RECONCILIATION.json and scalar_window7_reconciliation.npz for final scientific numbers. Original RESULT.json scalar-window1 outputs are explicitly unqualified and superseded.','',
 '## Provenance and original reproduction','',
 'Actual downloaded original full distributions reproduce archived heat moments and heat reconstruction to floating-point roundoff. The scalar CSV was generated with boxcar7 on J,u,K, not script default1; this exact contract is independently inferred and reproduced. The initial window1 mismatch was diagnosed and retained rather than concealed.','',
 '| Mach | Heat max abs across stored columns | Scalar reconstructed Delta max abs | Legacy R two-budget error | Corrected same-mask R error | Corrected scalar error |',
 '|---|---:|---:|---:|---:|---:|']
for m in [2,3,5]:
    r=json.loads((HERE/f'BGK_M{m}_regenerated/CONTRACT_RECONCILIATION.json').read_text())
    c=r['corrected_table_metrics_window7']['stored_centre']['fixed_legacy'];old=r['legacy_table_exact_mask_metrics']
    h=max(v['max_abs'] for v in r['heat_CSV_reproduction'].values());d=r['scalar_CSV_reproduction']['Delta_rec_max_abs']
    summary.append({'Mach':m,'heat_max_abs':h,'scalar_Delta_max_abs':d,'legacy_Rtwo':old['R_budgetDelta_relL2'],
       'corrected_stored_Rtwo':c['R_budgetDelta_relL2'],
       'corrected_macros_Rtwo':r['corrected_table_metrics_window7']['recomputed_macros']['fixed_legacy']['R_budgetDelta_relL2'],
       'target_errors':r['target_moment_errors_vs_requested_macros'],'source_anchor':r['source_and_anchor_diagnostics'],'input_sha256':r['input_sha256']})
    lines.append('| {} | {:.3g} | {:.3g} | {:.6f}% | {:.6f}% | {:.6f}% |'.format(m,h,d,100*old['R_budgetDelta_relL2'],100*c['R_budgetDelta_relL2'],100*c['Delta_budget_relL2']))
    for key in r['corrected_table_metrics_window7']:
        assert np.isfinite(list(r['corrected_table_metrics_window7'][key]['fixed_legacy'].values())).all()
lines+=['','## Meaning of the corrections','',
 'Corrected heat source is q[Md]-q[f] with computational tau1 in x_mfp; anchor is actual Fq[0]; geometry is directly reduced at the specified centre. Corrected scalar source retains C4[Md], rather than assuming continuum15rhoT2. All archived plateau/mask contracts are explicit; both fixed legacy and new corrected-S masks are reported.','',
 'The source and scalar corrections do not remove finite-time/spatial-discretization errors. Their small effects do not establish stationary shocks. The large one-channel R error remains about63--64%, while two-channel recovery remains a few percent. This supports the channel-complement conclusion with revised quantitative diagnostics, not exact recovery or convergence.','',
 'Md is reconstructed in float64 with four bounded Newton iterations on the actual stored velocity grid/weights. Target mass/mean/temperature errors versus requested macros are recorded in each contract. Original float32/CUDA target identity is not asserted. Stored-centre and consistently recomputed-macro results remain separate; no silent recentering or normalization.','',
 'Fullstate/file hashes, archived-CSV reproduction, target checks, source/anchor sizes and correlations are in the machine-readable contracts. Distribution memmaps and originals are preserved locally; exclude huge extracted distribution.npy files from ordinary Git.']
(HERE/'BGK_FULLSTATE_CLOSURE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(HERE/'BGK_FULLSTATE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k not in ['target_errors','source_anchor']} for r in summary],indent=2))
