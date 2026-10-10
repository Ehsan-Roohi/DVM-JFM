# Independent heat-budget specification

Read-only equations/source audit, 2026-10-10. No solver, fullstate reduction or figure replacement performed here. Parent owns bounded reduction implementation.

## Exact finite-grid identities

Use the actual velocity weights and a single declared centralisation u(x). Define psi=.5*c²*cx, q=sum(w*psi*f), Fq=sum(w*vx*psi*f), sigma=sum(w*(cx²-c²/3)*f), m=sum(w*(cx³-3cx*c²/5)*f).

For a steady BGK equation vx*f_x=(Md-f)/tau, differentiating the moving moment kernel gives

    Fq_x + u_x*A = Qq,
    A=sum(w*vx*(.5*c²+cx²)*f),
    Qq=(q[Md]-q[f])/tau.

The sign is important: reconstruct Fq_prime=Qq-u_x*A. The original script uses -q/tau-u_x*A, omitting q[Md]/tau. Matching mass/momentum/energy on a finite velocity grid does not impose zero heat flux of Md.

For moments evaluated at the exact same centre, A=u*(2.5*rho*T+sigma)+m+11q/5. In the archived CSV, moments are evaluated at ux_file but rho/T from the file differ slightly from moments of f. Therefore the exact algebraic reconstruction of A available from the CSV uses ux_file*(2.5*rho_from_f*T_from_f+sigma)+m+11q/5. The script defines T_from_f=C2/(3*rho_from_f) at ux_file, not at a different independently recentered velocity. Retained geom_term uses rho_file*T_file and a smoothed velocity derivative. These alternatives must be reported rather than silently mixed.

For the continuum-reference convention used in the archived CSV:

    Delta_c=C4[f]-15*rho_file*T_file²,
    R_c=Btr[f]-7*T_file*sigma,
    S_c=R_c+Delta_c/3,
    Fq=u_file*q+.5*(5*rho_file*T_file²+7*T_file*sigma+S_c).

This algebraic identity holds on any velocity grid, regardless of quadrature defects, because its reference terms cancel. It does not justify using a continuum-Maxwellian source.

For the manuscript's discrete-reference convention:

    Delta_d=C4[f-Md],
    R_d=Btr[f-Md]-7*T*sigma[f],
    S_d=R_d+Delta_d/3,
    Fq=u*q+.5*(Bxx[Md]+7*T*sigma[f]+S_d),
    Bxx[Md]=sum(w*c²*cx²*Md).

At f=Md, sigma[Md] need not vanish and R_d=-7T*sigma[Md]. Thus S_d=0 is not an admissible assumed equilibrium anchor. If removing display offsets o_R and o_D, subtract their combined o_R+o_D/3 consistently in the reconstructed S. Raw collision Qdelta=-Delta_d/tau; corrected displayed Delta requires the corresponding offset accounting.

## Anchor and coordinates

Actual boundary data determine incoming populations only. The appropriate saved-profile anchor is Fq_direct[0] at the very first grid point x[0], followed by cumulative trapezoid integration from exactly that point. The original mean of 2.5*rho*T² over the first nx//30 points is neither the true first-point flux nor a full boundary-distribution constraint. Report its difference from Fq_direct[0]. Do not use a mean over an interval but attach it to x[0] without adjustment.

The frozen BGK solver advances in x_mfp with dt=CFL*dx/vmax and collision multiplier exp(-dt), hence computational tau=1. It writes x_scaled=x_mfp/(2L) and kn_eff=1/(2L). A budget differentiated in x_mfp needs tau=1; differentiated in x_scaled needs coefficient1/kn_eff=2L. Choosing x_mfp and kn_eff together is a normalization error. The archived CSV q_over_kn equals q for the available cases, consistent with explicit kn=1 in x_mfp. Parent should verify this equality numerically and reject absent/contradictory metadata.

## Frozen target algorithm and provenance

Read-only paper root: C:/Users/roohie/OneDrive - University of Massachusetts/Documents/ChatGPT/ریوایز جی اف ام tail

- revision_runs/source/dvm_bgk_normal_shock_conservative_hmom_densemicro.py imports conservative_discrete_maxwellian from dvm_bgk_normal_shock_conservative.py; original high-moment diagnostic hard-codes niter=4 and evolution defaults corr_iters=4.
- Base target algorithm fits exponent parameters u_p,logT_p with two-variable Newton to mean/temperature, rescales density, uses targetT floor1e-8, det floor1e-14, du and dlogT clipped to[-.25,.25], final Md floor1e-30. The Newton iteration count is finite, not exact moment matching by assertion. Original evolution default is float32.
- Velocity grid is Cartesian linspace[-vmax,vmax] with tensor-product trapezoid endpoint half weights, indexing ij.
- revision_runs/model_audit/evidence/4.json identifies original baseline fullstates under /project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep/: standing_M2_hmom_x40_nx1600_v97_19_19_vmax12_fullstate.npz; standing_M3_hmom_x60_nx1800_v121_23_23_vmax16_fullstate.npz; standing_M5_hmom_x80_nx2200_v141_27_27_vmax22_fullstate.npz.
- These exact M2/M3/M5 fullstates were not found in the local paper-root file inventory. Do not substitute newer M5 sensitivity or Mach6/8/12 distributions. Remote archival input identity still needs hashes/config binding.
- Archived moments are revision_runs/model_audit/evidence/5.csv,6.csv,7.csv, corresponding to appendix_validation_dvm/exact_heatflux_budget/M{2,3,5}_exact_budget_xmfp_exact_heatflux_budget_profiles.csv; figure1_redesign_20261007/inputs contains named copies.
- Original heat-budget source is revision_runs/model_audit/source_snapshot/src/moment_closure_next/exact_heatflux_budget_from_fullstate.py. Its corrected normalization-guard version is in corrected_20261001; that correction was not a fullstate regeneration.

Read source SHA256 bindings: base DVM `d7a73766437091d9a376761ebf7ebf358c3b396f69e368069d89b4711e1d33f8`; hmom wrapper `855fd0a909f2e1e225d8b74a98a85fbc1fad720d991961907ebe9ee722dd892c`; archived heat postprocessor `628e1067c25740ed8e19c009b06e56aca24ea52bc3fbc6947d315a4a1e16dc37`. They bind this audit to source contents, not independently prove original run-source identity.

## Bounded correction scope and necessary fixtures

The CSV is sufficient for a source/anchor correction conditional on a specified reconstructed Md and grid, without loading f. Preserve the original continuum R/Delta and centralisation in this diagnostic. It is not proof of exact original distribution/target identity or steady convergence. Reconstructing Md from file fields and from_f fields are different operations. If converting moments to centre u_from_f, transform all central moments consistently or regenerate from f; do not only change u.

Compare float64/4, float64/12 and, when feasible, original float32/4 target reductions; report conservation mismatch and qMd sensitivity. Record binding uncertainty rather than label any reconstruction exact historical Md without receipts.

Required cheap algebra/negative fixtures:

1. A synthetic positive f on an asymmetric/truncated weighted velocity grid verifies direct Fq equals continuum and discrete decompositions; Bxx[Md] cannot be replaced by5rhoT² when it differs.
2. f=Md on a finite grid verifies Qq=0 while qMd may be nonzero; -q/tau alone must fail this control.
3. A constructed Fq profile with known Qq and geom verifies integration sign and first-point anchor exactly; shifted or interval-mean anchor must fail.
4. Coordinate rescaling x_scaled=x_mfp/(2L) preserves the integrated reconstructed Fq only when collision/geometric scaling transforms consistently; mixed x_mfp/kn_eff must fail.
5. A plateau-offset control verifies identical raw physical Qq before display changes and consistent S offset afterward.
6. No norm of a newly reconstructed channel should be represented as improved steady-state residual without separately evaluating stationarity and streaming discretization residual. Continuum finite-difference steady balances need not be exactly satisfied by finite-time first-order upwind/split states.

No upstream/fullstate completion claim follows from passing these moment-level fixtures.
