"""Use corrected results in the article; keep version comparisons in Git audit."""
from pathlib import Path
import json,shutil
F=Path(__file__).resolve().parent;P=F.parent/'publication';file=P/'manuscript/main.tex'
s=file.read_text(encoding='utf-8')
paragraph='The archived metrics for this check record '
start=s.index(paragraph);end=s.index('\n\\end{revision}',start)
s=s[:start]+r'''The check uses $Pr=2/3$ and a unit computational source scale. The distribution minimum is near $10^{-30}$, while the signed target minimum is approximately $-9.96\times10^{-5}$; the collision projection therefore does not certify a nonnegative Shakhov target or a stationary shock. Each displayed reference and reconstructed channel has the mean of the first and last $8\%$ of points removed, with active mask $|S|>0.02\max|S|$. The agreement in the scalar channel is plateau-corrected: $\Delta_{\rm source}+\Kn Q_\Delta$ is a constant offset, approximately $0.06816$. The flux-side subtraction uses $5\rho T^2$, whereas the directly evaluated closure moments use $f-M_d$, so the small flux-channel discrepancy also contains the discrete Maxwellian reference defect. The numerical curves have been regenerated from the archived full distribution; the input, code, profiles and provenance are deposited with the article.'''+s[end:]
method=r'''
The numerical budgets use the actual discrete velocity weights and retain the Maxwellian target moments in the collision contractions. In particular, $Q_q=(q[M_d]-q[f])/\tau$ and the scalar reconstruction is
\begin{equation}
\Delta_{\rm bud}=\sum_k w_k|\boldsymbol c_k|^4M_{d,k}-15\rho T^2-\tau\left(\frac{\mathrm dJ_\Delta}{\mathrm dx}+4\frac{\mathrm du}{\mathrm dx}K\right).
\end{equation}
Macroscopic fields are recomputed consistently from the stored distribution, and the conservative discrete target is evaluated in double precision with four matching iterations. The heat-flux integral is anchored by its directly evaluated value at the first grid point. With the stored \texttt{x\_mfp} coordinate, computational $\tau=1$ is used. Derivatives employ one $(1,4,6,4,1)/16$ pass on $u$ for the heat budget and a seven-point boxcar on $J_\Delta,u,K$ for the scalar budget. Both channels retain the first/last $8\%$ plateau subtraction; errors use the documented $2\%$ active mask. These budget checks do not establish stationarity or independent grid convergence.
'''
anchor=r'No direct $R^{\cl}_{xx}$ anchoring or supervised $R$ data enter this reconstruction.'
assert anchor in s;s=s.replace(anchor,method+'\n'+anchor,1)
oldrows=['M2 & 3.834 & 64.362 & 3.356 & 4.090 & 0.99960\\\\','M3 & 3.435 & 63.226 & 3.102 & 3.537 & 0.99954\\\\','M5 & 2.151 & 63.113 & 2.219 & 2.428 & 0.99975\\\\']
updated=[]
for m,old in zip((2,3,5),oldrows):
    c=json.loads((F/f'BGK_M{m}_regenerated/CONTRACT_RECONCILIATION.json').read_text())
    r=c['corrected_table_metrics_window7']['recomputed_macros']['fixed_legacy']
    values=[100*r[k] for k in ('Delta_budget_relL2','R_delta0_relL2','R_fullDelta_relL2','R_budgetDelta_relL2')]
    row=f'M{m} & '+' & '.join(f'{v:.3f}' for v in values)+f" & {r['R_budgetDelta_correlation']:.5f}\\\\"
    assert old in s;s=s.replace(old,row);updated.append(100*r['R_budgetDelta_relL2'])
old='The errors drop to $4.09\\%$, $3.54\\%$ and $2.43\\%$'
new='The errors drop to '+', '.join(f'${v:.2f}\\%$' for v in updated[:2])+f' and ${updated[2]:.2f}\\%$'
assert old in s;s=s.replace(old,new)
old='A complete raw-distribution deposit and regeneration of the legacy budget/source-fixed inputs remain outstanding.'
new=r'The archived budgets have been regenerated with their documented discrete-source contracts. The full distribution used for Figure~\ref{fig:shakhov_check} is included in the public repository with its hash; a complete public deposit of the larger BGK raw distributions remains outstanding.'
assert old in s;s=s.replace(old,new)
file.write_text(s,encoding='utf-8')
history=P/'audit/reproducibility_followup/historical_figures';history.mkdir(parents=True,exist_ok=True)
target=P/'manuscript/figures/fig3_two_budget_recovery.pdf'
shutil.copy2(target,history/target.name)
shutil.copy2(F/'corrected_two_budget_audit.pdf',target)
print('Updated article numbers, concise method and corrected figure3; historical comparison in audit only')
