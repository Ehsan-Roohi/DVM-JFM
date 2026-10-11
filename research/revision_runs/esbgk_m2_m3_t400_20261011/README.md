# Independently evolved ES-BGK Mach-2/3 shocks: frozen t=400 deposit

These are actual reduced-distribution kinetic evolutions, not source evaluations on BGK fields. Pr=2/3, alpha_ES=-1/2, mu=T, tau_ES=3/(2 rho). The relaxation-time factor is required to match the Shakhov viscosity. Geometry, incoming equilibrium reservoirs and axial velocity quadrature are specified in manifest.json. The g/h/k reduction retains the transverse contributions to the contracted fourth moments.

## Completed results and limits

Six completed cases are frozen here: M2/M3 baseline, coupled spatial/time refinement and subsequent velocity refinement. All reach approximately t=400. None passes the stationarity criterion. Budget errors reconstruct a steady-form balance on a nonstationary snapshot; they include transient and discretization effects and are not collision-model errors. Positivity and successful integration do not establish global stability. Separate half-step cases and later continuation results are excluded. Historical PLAN.md is preserved as the original preparation contract; continuation_*.json only records the separately requested continuation, not completed science.

## Reproduce budgets and figures

From this directory, with Python, NumPy and Matplotlib:

    python -m pip install -r requirements.txt
    OPENBLAS_NUM_THREADS=1 python reproduce.py

On Windows set OPENBLAS_NUM_THREADS=1 using the shell's environment syntax. The script checks checkpoint/configuration identities and saved file hashes, recomputes all six S/Delta/R budget errors from checkpoint populations, and regenerates both Mach comparison figures. Outputs are written under reproduced/. RECOMPUTATION_CHECK.json records the numerical check. Plot provenance records source identities and density-midpoint translations. DSMC profiles are raw, not smoothed; the Shakhov DVM and corrected R26 curves are distinct archived calculations. They do not provide an exact solution for ES-BGK.

## Run the kinetic solver

The archived esbgk.py, shakhov_reference.py and manifest.json are the exact frozen sources used for these cases. PASS.json binds their hashes. A fresh run requires a separate working copy without the deposited runs/ outputs; run `python esbgk.py --gate`, then `python esbgk.py --index 0` (indices 0 through 5 select the deposited cases). This is a CPU-intensive integration, not necessary to reproduce the deposited budgets and plots. Do not overwrite the archival checkpoints. Indices 6 and 7 are the separately planned half-step controls and are not completed results in this deposit.

The final scalar target excess uses the full reduced expression cx^4*g_G + 2*cx^2*h_G + k_G, rather than axial g_G alone. Spatial derivatives use numpy.gradient(edge_order=2); integration uses a cumulative trapezoid with the actual upstream heat-flux anchor. The common active mask is |S_direct| > .02*max|S_direct|. Relative L2 percentages use direct velocity moments on that mask and no independent curve alignment.

## Data integrity

DEPOSIT_MANIFEST.json lists SHA-256 and sizes of all deposited files except itself and Python caches. Checkpoints contain the complete g/h/k distributions used by this reduced solver; they are not sampled full three-dimensional f arrays. All six checkpoint/profile/budget hashes matched the original terminal STATUS.json records, and all budget errors reproduced to an absolute tolerance of 1e-12 on the dimensionless relative errors. This evidence concerns reproducibility, not stationarity.
