# Independent source review of chunked BGK budget regeneration

Scope: static mathematical review of `regenerate_bgk_budgets.py`; no duplicate full-distribution computation. Parent/other agent owns execution.

The direct velocity-centre geometry is correct:

`A = <vx(.5 c² + cx²) f> = u(2.5 rho T + sigma_xx) + mxxx + 2.2 qx`.

The heat budget uses `dFq/dx = Qq − u' A`, with `Qq = q(Md)−q(f)` for frozen tau1. Anchoring at actual upstream `Fq[0]` is explicitly distinguished from the historical equilibrium approximation; a constant anchor correction disappears under common plateau subtraction.

The central scalar forcing is correctly `dJ/dx + 4u'K`, because differentiation of `c^4` contributes `−4cx c² u'`. Thus `Delta_rec = C4(Md) − forcing −15rhoT²` preserves the discrete target fourth-moment defect. Treating that target term as identically zero would impose an unjustified continuum-Maxwellian identity.

The historical table convention matches the archived plotting source: subtract the mean of the two8%endpoint regions; choose the mask using `abs(S_reconstructed) >= .02*peak`; use the clean `M_delta_budget` CSV before the smooth7fallback; evaluate Delta and R variants on that same mask. New corrected-reconstructed masks are separately labelled, avoiding silent mask replacement.

Stored-centre and recomputed-macro reductions are separate. The centred temperature returned by `reduce` is valid when evaluated about its input centre; the recomputed branch feeds the recomputed velocity and temperature back consistently. The script does not invent an evolved/newly converged state.

One requested qualification addition: report `max|TMd−requestedT|` as well as target mass/mean errors. The four-iteration double-precision target is explicitly a diagnostic reconstruction; temperature mismatch would materially affect source corrections. Stored-macro deviations and target-macro constraint errors should remain separate. This review does not certify actual computed values or the historical CUDA target identity.
