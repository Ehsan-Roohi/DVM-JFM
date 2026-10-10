# Bounded fullstate preparation

`regenerate_bgk_budgets.py` is an isolated postprocessor, not a shock evolution code. Static parsing passed; fullstate computation has not been started by this agent. Parent decides actual resource/time envelope and independent package review before one case at a time.

It streams the compressed distribution member onto local writable audit storage and reads it by a read-only NumPy memmap. Each chunk uses the actual recorded flattened velocity coordinates and weights. Default four spatial rows; allowed maximum eight; estimated dense workspace is capped at1GiB. Native RAM must be at least4GiB initially and every extraction/chunk/Newton check must retain1GiB. No other program/process is modified. Remote execution is rejected until a separately approved Linux resource gate is supplied.

Two centralisations are deliberately separate: original stored rho/u/T, and all macros consistently recomputed from f. It reconstructs a double-precision four-iteration target on the actual grid. This target is conditional diagnostic Md, not bitwise original float32/CUDA history. The source is evaluated using qMd-q; the scalar source uses C4Md-C4f. Original archived heat moments and reconstructed budget are independently compared to exact CSV columns; original clean scalar C4/J/K and recovered Delta likewise compared. Coordinate arrays must agree.

Corrected heat reconstruction uses direct Fq at the first actual point, the raw integral A evaluated directly from f, and the source including target heat flux. Corrected scalar inversion retains C4Md rather than assuming15rhoT². Reports explicitly separate raw5-percent heat mask and plateau2-percent mask on the reconstructed S used by the archived two-budget plotting convention. Errors from different masks cannot be merged.

Remaining review points before any publication: double versus historical target arithmetic; clean versus smooth7 scalar source selected by figure provenance; fullstate hashes and exact original-run bindings; no stationary-shock inference from a finite-time profile's differential-budget residual.
