# Independent ES-BGK Mach 2 and Mach 3

Authorized by the human user on 2026-10-10. Independent ES-BGK shock fields,
not a target-source substitution on saved BGK fields.

Matched Shakhov contract: upstream rho=T=1, R=1, gamma=5/3; Mach-dependent
Rankine-Hugoniot equilibrium incoming reservoirs; mu=T (omega=1, mu1=1);
Pr=2/3; v in [-40,40], trapezoidal quadrature; L=30 (M2),45 (M3).
ES alpha=1-1/Pr=-1/2 and tau_ES=(1-alpha)*mu/p=1.5/rho.
Using tau_ES=mu/p would fail viscosity matching.

Conservative first-order upwind finite volume; implicit self-consistent BE
collision stress, with analytically integrated transverse Gaussian target.
Three reduced distributions g,h,k retain all required axial fourth moments.
Finite axial target errors are measured, no clipping or silent normalization.

For each Mach: baseline dx=.1875,dv=.2; space dx=.09375,dv=.2;
velocity dx=.09375,dv=.1; time dx=.09375,dv=.2,CFL=.4 rather than .8.
Final existing Shakhov finest dx=.046875 is finer than these initial ES tests.
These initial grids therefore do not establish matched-grid error against it.
All tests start from the same analytic reservoir mixture, tmax=400, at most
21000 wall seconds per packet, max two simultaneous single-CPU jobs.
No automatic resume or tolerance relaxation. Checkpoints retain exact hashes.

Stationarity requires five consecutive physical-time-2 comparisons with
all eight family changes <1e-7 AND full-step moment RHS <1e-7.
Also report independent final-post RHS, face mass/momentum/energy flux spans,
positive state and reduced distributions. tmax alone does not pass.

Direct S,R,Delta are velocity moments. Heat integration uses actual upstream
F anchor, finite-grid target q, and u' Lq. Scalar reconstruction uses
the actual ES finite-grid target Delta_G (continuum 3 alpha^2 sigma^2/rho)
and derivative of J plus 4u'K. R_rebuilt=S_rebuilt-Delta_rebuilt/3.
Record relative active-zone L2 errors, differential balances, grid/time/velocity
changes. Active mask fixed abs(S_direct)>2%max(abs(S_direct)). These steady-form
budgets must not be interpreted as validation on a transient profile.

Live I/O: allocated scratch workspace, expiry Nov 6 verified before submission;
local independent source copy here. No /project live writes. After terminal
closure preserve verified scratch plus local checkpoints/provenance/results.
Scientific acceptance is separate from job exit, algebra gate and tmax.
