# Numerical methods and reproduction of Figures 14 and 15

Run `python plot_figures.py` (NumPy, Matplotlib) in this directory to reproduce
the two PDF figures from the supplied CSV profiles. The figure titles identify
physical methods and parameters; numerical evaluation times and diagnostics are
recorded in PROVENANCE.json. ES-FP Mach-4/5 profiles correspond to t=29.0625
and 12.1774475098, respectively. Their runs continue separately.

## Common kinetic discretization
The three arrays g,h,k are transverse velocity integrals weighted by 1,
r_perp^2,r_perp^4. Axial velocity quadrature retains all moments plotted here.
Space uses first-order conservative upwind finite volumes, a shared flux per
face and equilibrium incoming Rankine-Hugoniot reservoirs. Each case's L,nx,nv,
vmax,CFL,Pr and viscosity are recorded in its manifest or figure provenance.
All curves use one density-midpoint translation, no shock-width fitting and no
smoothing. R is the trace-free fourth moment after subtracting 7*T*sigma_xx;
Delta is the fourth scalar moment minus 15*rho*T^2; S=R+Delta/3.

| Method | Pr and viscosity matching | Collision / solution algorithm |
|---|---|---|
| BGK | Pr=1, mu=T, tau=1/rho | analytic exponential collision; native C transport/collision checked against Python |
| Shakhov DVM | Pr=2/3, mu=T, tau=1/rho | analytic homogeneous collision, heat-flux decay exp(-Pr*dt/tau); reduced transverse target integrals |
| ES-BGK | Pr=2/3, alpha=-1/2, mu=T, tau=3/(2*rho) | backward Euler with implicit post-collision stress and anisotropic Gaussian target |
| ES-FP | beta=-5/4 on Pr=2/3 branch; tau=2*(1-beta)/rho, mu=T | implicit conservative Chang-Cooper axial velocity fluxes and sequential tridiagonal solves; transverse production retained |
| Shakhov R26 | Pr=2/3, mu=T | mixed steady balance/closure residual solve, continuous piecewise quartics, 8 Gauss points per element, sparse augmented Levenberg-Marquardt |
| Bird DSMC | model-derived Pr, VHS omega=1 | particle streaming and stochastic binary collisions; 2000 particles/cell, dt=.005, t=400, averages over t=200--400 |

The ordinary BGK operator does not become Pr=2/3 by changing tau. The figure
therefore correctly identifies its Pr=1 result.

## Source map
- `solvers/shock_matrix_tol6_20261011/`: BGK native and Python implementations,
  ES-BGK and reduced Shakhov references, matrix driver, cases and gates.
  Compile the native kernels using the flags in compile.sbatch before running
  the corresponding gate. These files are the production sources, not a new
  simplified numerical method.
- `solvers/esfp_fei_gatefix_20261012/`: exact adaptive ES-FP Mach-4 source.
  Finite-grid drift/diffusion fitting is bounded and measured.
- `solvers/esfp_m5_entropy_production_20261012/`: exact Mach-5 ES-FP source.
  The physical drift/diffusion are fixed. A positive exponential relative-entropy
  moment projection enforces BE mass/mean/axial variance, with both L1 and
  second-moment corrections strictly below 2%. No population clipping.
  Validation and ancestry records bind the actual production files. Recompiled
  libraries must pass gates and be rebound before continuation; do not edit old
  hash receipts to bypass validation. Continuation expects a matched checkpoint.
- R26 full equations, repaired tensor projections and numerical algorithm:
  `../revision_runs/r26_M3_shakhov_20261009/` (especially `stf_repaired_tensor/`).
- DSMC source modifications and fourth-moment diagnostics:
  `../bird_dsmc_modifications/` and
  `../revision_runs/bird_dsmc_m2_m5_high_particles_20261009/`.
  The original Bird dependency and how to obtain it are specified there;
  it is not redistributed without a source licence.

ES-FP uses zero axial velocity endpoint fluxes. The adaptive beta bound from
Fei et al. is applied when diffusion would lose admissibility; this changes local
effective Pr to 3/[2*(1-beta)]. The plotted states have effective Pr=2/3.
Nonnegative reduced populations, positive density/temperature/covariance and
mass/momentum/energy balance remain mandatory. The Mach-5 kernel additionally
checks g*k >= h*h (within its stated floating-point tolerance).

## Meaning of residuals
For kinetic solutions the eight-field rate is max over fields and positions of
abs(Y_after_one_full_step-Y)/(dt*max(1e-8,max_x abs(Y))). It is distinct from
profile error against DSMC and from spatial/velocity/time accuracy. R26 uses
its own native tensor-equation and boundary/phase residuals. DSMC sampling
window changes are statistical diagnostics; a particle count alone is not a
stationarity or uncertainty certificate. Numerical metadata stays outside figure
titles. See DSMC_delta_audit.json for the stored Delta identity checks.

## References
Fei et al., AIAA J. 58, 2596--2608 (2020), doi:10.2514/1.J059029.
Gu & Emerson, JFM 636, 177--216 (2009), for the R26 tensor framework.
The current manuscript spells out the Shakhov specialization and tensor repairs.

The retained original owners are also supplied explicitly in solvers/bgk_fei_m2_wide_v1, solvers/bgk_fei_m3_m5_native_v1, solvers/esfp_m2_matched_v1 and solvers/shock_longtime_20261011/matched_matrix_v1. Their manifests and source gates distinguish the earlier native trajectories from the general matrix driver.
