# V22 notation audit (read-only scientific review)

Read the complete main.tex, all main equations/captions and appendices, and the three table fragments. The fragments are not included by main.tex (no input command); avoid reviving their historical budgets. No scientific outputs changed.

## Highest-priority inconsistencies

1. **Theta is redefined incompatibly**: ES Gaussian target covariance in eq:es_gaussian vs actual distribution covariance in the ES-FP admissibility paragraph. Use `\Theta` only for target covariance; actual covariance `\mathsf C=\rho^{-1}\int\boldsymbol c\boldsymbol c f\,\dd\boldsymbol v=TI+\sigma/\rho`. Replace FP paragraph with `D/(2\nu)=(1-\beta)TI+\beta\mathsf C`; every eigenvalue of `\mathsf C` below `9T/5`. This also clarifies the appendix's “full temperature tensor”.
2. **q/u/K aliases remain**: two_budget paragraph `Q_q=(q[M_d]-q[f])/\tau` -> `Q_q=(q_x[M_d]-q_x[f])/\tau`. Same paragraph smoothing inputs `u` -> `u_x`, `J_\Delta,u,K` -> `J_\Delta,u_x,K_\Delta`. Shakhov `q_S/q` -> `q_x[M_S]/q_x[f]`, and `Q_q/q` -> `Q_q/q_x[f]`, introduce `M_S` as Shakhov target. Appendix profiles `(rho,u,T,...)` -> `u_x`; `Y` is not defined in Mach8 stopping criterion (insert definition below).
3. **Scalar reference convention**: new eq5.19 correctly reconstructs continuum-reference Delta_c BEFORE plateau subtraction; old sentence “The discrete scalar budget reconstructs Delta_d” can look contradictory. Change to “The raw discrete-source inversion gives Delta_d; adding the explicit target-equilibrium offset in the displayed equation gives Delta_c.” General `Delta` must remain generic in rank theorem and model-law derivation, but local experiments should explicitly identify continuum/discrete/plateau variant.
4. **R vs Rcl**: consequence section uses `R_ij` without superscript and introductory hierarchy `R_ij`; change those moment variables to `R^{\cl}_{ij}` throughout. Leave gas constant R distinct and explicitly define it. Statistical `c` in error table is distinct from peculiar-velocity c; relabel correlation `c_e`.
5. **Mixture weights vs quadrature weights**: appendix isotropic example uses w_k,T_k already quadrature aliases. Use `a_\ell,T_\ell` positive mixture fractions, `\sum a_\ell=1`, so `T=\sum a_\ell T_\ell`, `Delta_c=15rho[\sum a_\ell T_\ell^2-T^2]`.
6. **Mean-free-path labels**: high-Mach table/caption still says upstream mean-free-path although BGK section identifies it as relaxation length. Replace “upstream mean-free-path coordinate” with “upstream relaxation-length coordinate”. This is substantive consistency, not typography.

## Exact compact insertions at first use

### Before table:solver_campaigns (end opening Section2 paragraph)
```latex
We use $Pr=\mu c_p/\kappa$, where $\mu$ is the dynamic
viscosity, $\kappa$ the thermal conductivity and $c_p=5R/2$
the monatomic specific heat at constant pressure. Here $R$
is the specific gas constant, set to one in computational units;
$R^{\cl}_{ij}$ instead denotes a fourth-order closure moment.
The upstream Mach number is $M_1=u_1/\sqrt{\gamma RT_1}$,
with $\gamma=5/3$. Subscripts 1 and 2 denote upstream and
downstream reservoir states, respectively, and $p=\rho RT$
is the equilibrium scalar pressure.
```

### After peculiar velocity first definition near eq:q_sigma
```latex
The coordinate $x$ is normal to the shock, $\boldsymbol v$ is
the molecular velocity, and $\boldsymbol e_x$ is the unit vector
in that direction; $v_x$ and $c_x=v_x-u_x$ are their streamwise
components. The distributions are mass-weighted, so
$\rho=\int f\,\dd\boldsymbol v$,
$\rho u_x=\int v_x f\,\dd\boldsymbol v$, and
$3\rho T=\int |\boldsymbol c|^2f\,\dd\boldsymbol v$ in units $R=1$.
Tensor indices range over $x,y,z$; repeated Cartesian indices
are summed, and $\delta_{ij}$ is the Kronecker delta.
The full stress is
$\sigma_{ij}=\int(c_ic_j-|\boldsymbol c|^2\delta_{ij}/3)f\,\dd\boldsymbol v$.
```
For introductory m_ijk, define only if retained as a variable: `m_{ijk}=\int c_{\langle i}c_jc_{k\rangle}f\,\dd\boldsymbol v`, where brackets denote the symmetric trace-free part. It is not used in numerical reconstruction, so a brief prose definition also suffices.

### Around table:high_mach_grids / first discretisation
```latex
$N_x$ and $N_{v_i}$ count spatial points and velocity nodes,
$L$ is the spatial half-length, and $v_{\max}$ the velocity-box
half-width. The symbols $\Delta x$, $\Delta v_i$ and $\Delta t$
denote their spacings and the time step. The CFL number
$C_{\rm CFL}=v_{\max}\Delta t/\Delta x$ controls explicit streaming;
$N_t$ denotes the completed number of time steps.
```
“H” domain half-length appears separately in metadata; say H=L for symmetric intervals only, not Mach8 asymmetric.

### After eq:null_transform and auxiliary-channel equation
```latex
Here $\eta=\eta(x)$ is a scalar perturbation with the same units
as $R^{\cl}_{xx}$, and $\mathcal N(\mathcal H)$ denotes the null
space of the observation operator. The coefficients $a$ and $b$
are supplied scalar weights defining the auxiliary observation
$\mathcal P$.
```
At heat integral: `x_0` is anchor, `xi` integration coordinate; primes in `u_x'` denote spatial derivatives. These are standard but useful due DSMC primes exception.

### Before/after eq:es_gaussian
```latex
The symbols $\det$ and $\tr$ denote determinant and trace,
respectively; $\sigma$ denotes the matrix with entries
$\sigma_{ij}$. The dimensionless coefficient $\alpha_{\rm ES}$
sets the anisotropic ES-BGK target covariance.
```
Replace ambiguous “up to sign convention” with exact current stress definition `\Lambda_ES=alpha_ES sigma/rho` (stress defined above; no remaining sign ambiguity).

### Following FP operator / linear-drift paragraph
```latex
$A_i$ is the velocity-space drift and $D_{ij}$ the diffusion
tensor. The positive rate $\nu$ sets the collision time scale,
and $\beta$ is the dimensionless diffusion-anisotropy coefficient.
The test function is $\phi(\boldsymbol c)=|\boldsymbol c|^4$.
```
Wick brackets: define `<...>` here as expectation under normalized Gaussian GTheta/rho (not spatial average).

### Under eq5.19
Current long insertion already defines k,wk,ck,Mdk,fk,rho,ux,T,tau,J,K correctly. Keep it, but avoid unnecessary restating all fields if reducing length. Label equation `eq:delta_continuum_budget`; current equation has no label. Define suffixes once:
```latex
The suffixes $d$, $c$, $\mathrm{bud}$, $\mathrm{full}$ and
$\mathrm{sparse}$ denote the discrete-equilibrium reference,
continuum-equilibrium reference, budget reconstruction, complete
scalar-field input and interpolated probe field, respectively.
```
Delta displayed must explicitly be `Delta_c-oDelta`; R displayed `Rcl_c-oR`; S displayed corresponding sum. Do not rename formulas without enforcing the offsets.

### After error formula
```latex
The discrete norm is
$\|g\|_{2,\mathcal A}=(\sum_{i\in\mathcal A}g_i^2)^{1/2}$;
$i$ indexes spatial samples. On the uniform comparison grids,
common quadrature factors cancel from relative errors.
The symbols $g_{\rm rec}$ and $g_{\rm ref}$ designate the reconstructed
and reference values of the chosen field, respectively.
```
Table eps_Delta,eps_R,rho_R follow this already-defined generic metric, so need no extra unrelated definitions. “Full Delta reference” means insert complete reference Delta into fixed reconstructed S, NOT independent exact R solution.

### DSMC after eq:dsmc_qdelta (replace existing short where paragraph)
```latex
Here $j$ labels a collision cell of volume $V_j$;
$\sum_{\rm coll\in j}$ runs over accepted binary collisions
accumulated during the exposure $\Delta t_{\rm samp}$.
The constant statistical weight $w_p$ is the mass represented
by a simulation particle (equivalently a molecular-count weight
when the computational molecular mass is one).
The pair labels 1 and 2 refer to the two colliding particles.
Their peculiar velocities use the same instantaneous
pre-collision cell mean $\boldsymbol u_j$:
$\boldsymbol c_\ell=\boldsymbol v_\ell-\boldsymbol u_j$ and
$\boldsymbol c'_\ell=\boldsymbol v'_\ell-\boldsymbol u_j$.
Primes here denote post-collision values, whereas $u_x'$
denotes a spatial derivative. The exposure in the denominator
is the time represented by the accumulated collision events,
not simply the spacing between stored moment samples.
The quantity $Q(f,f)$ is the Boltzmann binary-collision operator.
```
**Parent verify source implementation before saying “all physical timesteps”**: safest exposure definition above. Same for `Vj=Delta xj` unitarea and `wp` molecular count; need code confirm actual nondimensional mass convention.

### After DSMC protocol paragraph
```latex
For this hard-sphere campaign,
$\lambda_1=(\sqrt{2}\,n_1\Sigma_T)^{-1}$ is the upstream
mean free path, where $n_1$ is molecular number density and
$\Sigma_T$ the total hard-sphere collision cross-section.
Particles per cell (PPC) is the nominal upstream simulation-particle
count in a spatial cell. Independent seeds specify independent
pseudorandom initialisations; the 11001 stored samples are per
realisation, not independent collision events.
```
Insert exact `n1=1,SigmaT=1/sqrt2` ONLY after source confirmation, not from manuscript alone. Add after DSMCbudget “JDelta and KDelta are eq:J_K_delta fluxes evaluated from sampled particle distribution; these moments use continuum-reference Delta_c.”

### DSMC figure / statistics appendix
Figure Main_DSMC_high_moments `n=2 crossing` -> `rho=2 crossing` if mass unit1, OR explicitly define n=rho/m and explain n1=1 then crossing2. Mach3 downstreamrho3 makes midpoint2 correct.
```latex
A bar denotes the mean over the eight seed profiles. The suffixes
$c$ and $o$ identify corrected and original samplers, and
$s(g_c-g_o)$ is the pointwise sample standard deviation of their
eight matched-label differences.
```
Crossmethod errors before signed table:
```latex
$e_R=R^{\cl}_{xx,\rm model}-R^{\cl}_{xx,\rm DSMC}$,
$e_{\Delta/3}=(\Delta_{\rm model}-\Delta_{\rm DSMC})/3$,
and $e_S=S_{\rm model}-S_{\rm DSMC}$;
$\langle a,b\rangle=\sum_i a_ib_i$ on the comparison interval.
The coefficient $c_e$ is their normalized inner product.
```

### Crossmethod appendix reduced distributions and transport
Definitions g,h,k already present: preserve. Replace u by ux in savedfield list. Define `omega` viscosity exponent via `mu~T^omega`; rename VSS alpha to `alpha_VSS` avoiding confusion with ES coefficient. Define starred quantities as divided by corresponding upstream reference values. μref in Mach8 is upstream viscosity coefficient in μ=μref T^omega, not rate. Before Mach8 stationarity criterion define `$Y$ denotes each field in the stated five-field list`.

## Already defined / should not duplicate
T and I now correctly defined by recent insert. JDelta/KDelta already explicit before eq5.19. Qq,QDelta explicit model integrals. rho_g/eps_g defined in Section5 and govern tables. x_s defined Section2. Reduced g/h/k defined crossmethod appendix. sigma offdiagonal reduction and deltaij proof AppendixG, but early definition still needed before first ES equation. All included equation references appear to correspond to existing labels; eq5.19 new unlabelled display is the sole new reference opportunity.

## Notation not worth adding unsupported formulas
RBF ridge widths/noise distributions belong to actual interpolation contract; Section6 main archived test describes method but not full kernel/hyperparameters, AppendixFigure caption defines parameter grid for separate core-gap test. Do NOT silently apply those parameters to main archived sparse run. If parent has exact script, add Gaussian-RBF kernel and penalty definitions with actual ridge scaling; otherwise retain code citation without inventing numerics.
Math `so(3)` deserves small prose definition (“Lie algebra of three-dimensional rotations”) if retained. H-theorem, hyperbolicity etc are concepts, not additional numerical parameters.
