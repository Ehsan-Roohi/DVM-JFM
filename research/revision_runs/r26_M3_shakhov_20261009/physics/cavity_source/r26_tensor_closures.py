#!/usr/bin/env python3
"""Full-tensor Gu--Emerson nonlinear R26 closures for a planar 2D3V grid.

This module ports the tensor form used by the audited ASTR stage-3 v3 patch
(``2610137:R26_FULL_BULK_CLOSURE_V3_portable.patch``).  The implementation is
not component-expanded: all angular brackets are evaluated with explicit,
three-dimensional symmetric-trace-free (STF) projections.  This removes a
large class of x/y/z component and trace mistakes.

High-risk provenance note
-------------------------
The primary-paper nonlinear Eq. (25) contribution to ``psi`` contains

``54/7 STF(m[e,a,b] du[e]/dx[c]) + (8 - 6) m[a,b,c] div(u)``.

The primary-paper angle-bracket convention makes both the ``+8`` and ``-6``
terms scalar-divergence multipliers.  They are nevertheless retained as two
separate statements under ``equation25_mode='v3-literal'`` so a future source
audit cannot silently change this historically error-prone row.  No
speculative contracted-deformation alternative is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Final

import numpy as np

from r26_state import StateTensors, planar_state_to_tensors, validate_planar_state


PHI_C1: Final[float] = 2.097
PHI_C2: Final[float] = 0.291
PSI_Y1: Final[float] = 1.698
PSI_Y2: Final[float] = 1.203
PSI_Y3: Final[float] = 0.854
CLOSURE_PROVENANCE: Final[str] = (
    "ASTR stage-3 v3 portable patch commit 2610137; Gu--Emerson (2009) "
    "Maxwell-molecule nonlinear R26 bulk closures"
)


@dataclass(frozen=True)
class R26ClosureCoefficients:
    """One source-locked nonlinear closure coefficient/equation mode."""

    mode: str
    C1: float
    C2: float
    Y1: float
    Y2: float
    Y3: float
    provenance: str


_CLOSURE_COEFFICIENTS: Final[dict[str, R26ClosureCoefficients]] = {
    "jfm2009": R26ClosureCoefficients(
        mode="jfm2009",
        C1=PHI_C1,
        C2=PHI_C2,
        Y1=PSI_Y1,
        Y2=PSI_Y2,
        Y3=PSI_Y3,
        provenance=(
            "Gu--Emerson JFM 636 (2009) / audited ASTR v3 convention: "
            "C1=2.097,C2=+0.291,Y1=1.698,Y2=+1.203,Y3=0.854"
        ),
    ),
    "asme2009-cavity": R26ClosureCoefficients(
        mode="asme2009-cavity",
        C1=2.097,
        C2=-0.291,
        Y1=1.82,
        Y2=-1.203,
        Y3=0.854,
        provenance=(
            "John--Gu--Emerson ASME HT2009-88293 driven-cavity Eq. (19)-(21): "
            "C1=2.097,C2=-0.291,Y1=1.82,Y2=-1.203,Y3=0.854"
        ),
    ),
}


def closure_coefficients(mode: str) -> R26ClosureCoefficients:
    """Return an immutable complete coefficient set; partial swaps forbidden."""

    try:
        return _CLOSURE_COEFFICIENTS[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unsupported complete R26 closure mode {mode!r}") from exc


def resolve_closure_mode(
    *,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> str:
    """Resolve one complete closure mode without silently overriding a case.

    Solver backends receive the selected mode through ``case`` whereas the
    point/grid closure APIs commonly receive it as an explicit string.  This
    helper keeps those two entry paths consistent and rejects conflicting
    selections instead of falling back to the JFM default in part of a run.
    """

    case_mode = None if case is None else getattr(case, "r26_closure_mode", None)
    if coefficient_mode is None:
        mode = "jfm2009" if case_mode is None else str(case_mode)
    else:
        mode = str(coefficient_mode)
        if case_mode is not None and mode != str(case_mode):
            raise ValueError(
                "explicit R26 closure mode conflicts with case.r26_closure_mode"
            )
    # Validate the complete coefficient set at the propagation boundary.
    closure_coefficients(mode)
    return mode


@dataclass(frozen=True)
class R26Gradients:
    """Physical gradients; the derivative index is the first tensor index."""

    rho: np.ndarray  # (..., d)
    velocity: np.ndarray  # (..., d, i) = d u_i / d x_d
    theta: np.ndarray  # (..., d)
    heat_flux: np.ndarray  # (..., d, i)
    sigma: np.ndarray  # (..., d, i, j)
    R: np.ndarray  # (..., d, i, j)
    m: np.ndarray  # (..., d, i, j, k)
    Delta: np.ndarray  # (..., d)


@dataclass(frozen=True)
class R26Closures:
    """The rank-4 phi, rank-3 psi, and vector Omega closures."""

    phi: np.ndarray
    psi: np.ndarray
    Omega: np.ndarray
    equation25_mode: str = "v3-literal"
    provenance: str = CLOSURE_PROVENANCE
    coefficient_mode: str = "jfm2009"


def _check_tensor(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(array, dtype=float)
    if a.shape[-len(shape) :] != shape:
        raise ValueError(f"{name} must end in {shape}, got {a.shape}")
    if not np.isfinite(a).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return a


def _symmetrize_last(tensor: np.ndarray, rank: int) -> np.ndarray:
    a = np.asarray(tensor, dtype=float)
    if a.ndim < rank or a.shape[-rank:] != (3,) * rank:
        raise ValueError(f"rank-{rank} tensor must end in {(3,) * rank}, got {a.shape}")
    lead = tuple(range(a.ndim - rank))
    offset = a.ndim - rank
    orderings = tuple(permutations(range(rank)))
    out = np.zeros_like(a, dtype=float)
    for ordering in orderings:
        out += np.transpose(a, lead + tuple(offset + i for i in ordering))
    return out / float(len(orderings))


def stf2_project(raw: np.ndarray) -> np.ndarray:
    """Project the last two indices onto symmetric trace-free rank 2."""

    sym = _symmetrize_last(raw, 2)
    trace = np.trace(sym, axis1=-2, axis2=-1)
    return sym - np.einsum("ij,...->...ij", np.eye(3), trace) / 3.0


def stf3_project(raw: np.ndarray) -> np.ndarray:
    """Project the last three indices onto symmetric trace-free rank 3."""

    sym = _symmetrize_last(raw, 3)
    trace = np.einsum("...iik->...k", sym)
    delta = np.eye(3)
    correction = (
        np.einsum("ij,...k->...ijk", delta, trace)
        + np.einsum("ik,...j->...ijk", delta, trace)
        + np.einsum("jk,...i->...ijk", delta, trace)
    )
    return sym - correction / 5.0


def stf4_project(raw: np.ndarray) -> np.ndarray:
    """Project the last four indices onto symmetric trace-free rank 4."""

    sym = _symmetrize_last(raw, 4)
    trace2 = np.einsum("...iikl->...kl", sym)
    trace4 = np.einsum("...kk->...", trace2)
    delta = np.eye(3)
    single = (
        np.einsum("ij,...kl->...ijkl", delta, trace2)
        + np.einsum("ik,...jl->...ijkl", delta, trace2)
        + np.einsum("il,...jk->...ijkl", delta, trace2)
        + np.einsum("jk,...il->...ijkl", delta, trace2)
        + np.einsum("jl,...ik->...ijkl", delta, trace2)
        + np.einsum("kl,...ij->...ijkl", delta, trace2)
    )
    double = (
        np.einsum("ij,kl,...->...ijkl", delta, delta, trace4)
        + np.einsum("ik,jl,...->...ijkl", delta, delta, trace4)
        + np.einsum("il,jk,...->...ijkl", delta, delta, trace4)
    )
    return sym - single / 7.0 + double / 35.0


def _coordinate_or_spacing(coordinate: np.ndarray | None, spacing: float, n: int, name: str) -> np.ndarray | float:
    if coordinate is None:
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(f"{name} spacing must be positive")
        return float(spacing)
    c = np.asarray(coordinate, dtype=float)
    if c.shape != (n,) or not np.isfinite(c).all() or np.any(np.diff(c) <= 0.0):
        raise ValueError(f"{name} coordinate must be a strictly increasing vector of length {n}")
    return c


def _grid_derivatives(
    field: np.ndarray,
    *,
    tensor_rank: int,
    x: np.ndarray | None,
    y: np.ndarray | None,
    dx: float,
    dy: float,
    edge_order: int,
) -> np.ndarray:
    if field.ndim < 2 + tensor_rank:
        raise ValueError("field is missing the (ny,nx) grid dimensions")
    ny, nx = field.shape[:2]
    if nx < edge_order + 1 or ny < edge_order + 1:
        raise ValueError(f"grid must have at least {edge_order + 1} nodes per direction")
    xarg = _coordinate_or_spacing(x, dx, nx, "x")
    yarg = _coordinate_or_spacing(y, dy, ny, "y")
    d_dx = np.gradient(field, xarg, axis=1, edge_order=edge_order)
    d_dy = np.gradient(field, yarg, axis=0, edge_order=edge_order)
    d_dz = np.zeros_like(field)
    derivative_axis = -(tensor_rank + 1)
    return np.stack((d_dx, d_dy, d_dz), axis=derivative_axis)


def finite_difference_gradients(
    state: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    edge_order: int = 2,
) -> R26Gradients:
    """Compute Cartesian 2-D finite-difference gradients of the full tensors.

    ``state`` must have shape ``(ny,nx,17)``.  z derivatives are exactly zero,
    as required by the 2D3V reduction.  NumPy's three-point boundary stencil
    is used by default so a linear manufactured field differentiates exactly
    even at the walls.
    """

    u = validate_planar_state(state)
    if u.ndim != 3:
        raise ValueError(f"state grid must have shape (ny,nx,17), got {u.shape}")
    if edge_order not in (1, 2):
        raise ValueError("edge_order must be 1 or 2")
    tensors = planar_state_to_tensors(u)
    args = dict(x=x, y=y, dx=dx, dy=dy, edge_order=edge_order)
    return R26Gradients(
        rho=_grid_derivatives(tensors.rho, tensor_rank=0, **args),
        velocity=_grid_derivatives(tensors.velocity, tensor_rank=1, **args),
        theta=_grid_derivatives(tensors.theta, tensor_rank=0, **args),
        heat_flux=_grid_derivatives(tensors.heat_flux, tensor_rank=1, **args),
        sigma=_grid_derivatives(tensors.sigma, tensor_rank=2, **args),
        R=_grid_derivatives(tensors.R, tensor_rank=2, **args),
        m=_grid_derivatives(tensors.m, tensor_rank=3, **args),
        Delta=_grid_derivatives(tensors.Delta, tensor_rank=0, **args),
    )


def rotate_gradients(gradients: R26Gradients, orthogonal: np.ndarray) -> R26Gradients:
    """Rotate/reflection-transform gradients, including their derivative index."""

    q = np.asarray(orthogonal, dtype=float)
    if q.shape != (3, 3) or not np.allclose(q @ q.T, np.eye(3), rtol=0.0, atol=5.0e-13):
        raise ValueError("orthogonal must be a 3x3 orthogonal matrix")
    return R26Gradients(
        rho=np.einsum("ad,...d->...a", q, gradients.rho),
        velocity=np.einsum("ad,bi,...di->...ab", q, q, gradients.velocity),
        theta=np.einsum("ad,...d->...a", q, gradients.theta),
        heat_flux=np.einsum("ad,bi,...di->...ab", q, q, gradients.heat_flux),
        sigma=np.einsum("ad,bi,cj,...dij->...abc", q, q, q, gradients.sigma),
        R=np.einsum("ad,bi,cj,...dij->...abc", q, q, q, gradients.R),
        m=np.einsum("ad,bi,cj,ek,...dijk->...abce", q, q, q, q, gradients.m),
        Delta=np.einsum("ad,...d->...a", q, gradients.Delta),
    )


def _leading_shape(tensors: StateTensors) -> tuple[int, ...]:
    return np.asarray(tensors.rho).shape


def _broadcast_mu(mu: float | np.ndarray, leading: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(mu, dtype=float)
    try:
        value = np.broadcast_to(value, leading)
    except ValueError as exc:
        raise ValueError(f"mu with shape {value.shape} cannot broadcast to {leading}") from exc
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise FloatingPointError("viscosity mu must be finite and positive")
    return value


def _validate_tensors_and_gradients(t: StateTensors, g: R26Gradients) -> None:
    leading = _leading_shape(t)
    expected = {
        "velocity": (3,),
        "heat_flux": (3,),
        "sigma": (3, 3),
        "R": (3, 3),
        "m": (3, 3, 3),
    }
    for name, shape in expected.items():
        a = _check_tensor(getattr(t, name), shape, name)
        if a.shape[: -len(shape)] != leading:
            raise ValueError(f"{name} leading dimensions do not match rho")
    for name in ("theta", "Delta"):
        a = np.asarray(getattr(t, name), dtype=float)
        if a.shape != leading or not np.isfinite(a).all():
            raise ValueError(f"{name} must have shape {leading} and be finite")
    if np.any(np.asarray(t.rho) <= 0.0) or np.any(np.asarray(t.theta) <= 0.0):
        raise FloatingPointError("rho and theta must be positive")
    grad_shapes = {
        "rho": (3,),
        "velocity": (3, 3),
        "theta": (3,),
        "heat_flux": (3, 3),
        "sigma": (3, 3, 3),
        "R": (3, 3, 3),
        "m": (3, 3, 3, 3),
        "Delta": (3,),
    }
    for name, shape in grad_shapes.items():
        a = _check_tensor(getattr(g, name), shape, f"gradient {name}")
        if a.shape[: -len(shape)] != leading:
            raise ValueError(f"gradient {name} leading dimensions do not match rho")


def closures_from_tensors(
    tensors: StateTensors,
    gradients: R26Gradients,
    *,
    mu: float | np.ndarray = 1.0,
    equation25_mode: str = "v3-literal",
    coefficient_mode: str = "jfm2009",
) -> R26Closures:
    """Evaluate the audited v3 nonlinear phi, psi, and Omega closures."""

    if equation25_mode != "v3-literal":
        raise ValueError("only the source-faithful 'v3-literal' Eq. (25) mode is implemented")
    coefficients = closure_coefficients(coefficient_mode)
    _validate_tensors_and_gradients(tensors, gradients)
    lead = _leading_shape(tensors)
    muv = _broadcast_mu(mu, lead)

    rho = np.asarray(tensors.rho, dtype=float)
    theta = np.asarray(tensors.theta, dtype=float)
    pressure = rho * theta
    q = np.asarray(tensors.heat_flux, dtype=float)
    sigma = np.asarray(tensors.sigma, dtype=float)
    rr = np.asarray(tensors.R, dtype=float)
    mm = np.asarray(tensors.m, dtype=float)
    delta_moment = np.asarray(tensors.Delta, dtype=float)

    gu = np.asarray(gradients.velocity, dtype=float)
    gs = np.asarray(gradients.sigma, dtype=float)
    gq = np.asarray(gradients.heat_flux, dtype=float)
    gr = np.asarray(gradients.R, dtype=float)
    gm = np.asarray(gradients.m, dtype=float)
    grho = np.asarray(gradients.rho, dtype=float)
    gtheta = np.asarray(gradients.theta, dtype=float)
    gdelta = np.asarray(gradients.Delta, dtype=float)

    div_sigma = np.einsum("...eae->...a", gs)
    div_q = np.einsum("...aa->...", gq)
    div_u = np.einsum("...aa->...", gu)
    # Primary notation: sigma_ml * d_l u_m.  Our gradient storage is
    # gu[derivative, component], whereas ASTR dvel[component, derivative].
    sigma_grad_u = np.einsum("...ab,...ba->...", sigma, gu)

    # Eq. (23), phi_ijkl.  Each raw product is projected in full 3-D.
    raw0 = np.moveaxis(gm, -4, -1) / rho[..., None, None, None, None]
    raw0 -= np.einsum("...abc,...d->...abcd", mm, grho) / rho[..., None, None, None, None] ** 2
    s0 = stf4_project(raw0)
    # sigma_ij d_l u_k and R_ij d_l u_k.  STF4 makes k<->l transposition
    # numerically equivalent, but keep the primary index orientation explicit.
    s1 = stf4_project(np.einsum("...ab,...dc->...abcd", sigma, gu))
    s2 = stf4_project(np.einsum("...abc,...d->...abcd", mm, div_sigma))
    s3 = stf4_project(np.einsum("...ab,...dc->...abcd", rr, gu))
    s4 = stf4_project(np.einsum("...ab,...cd->...abcd", sigma, sigma))
    phi = (
        -4.0 * muv[..., None, None, None, None] / coefficients.C1 * s0
        - 12.0 * muv[..., None, None, None, None] / (coefficients.C1 * rho[..., None, None, None, None]) * s1
        + 4.0 * muv[..., None, None, None, None]
        / (coefficients.C1 * pressure[..., None, None, None, None] * rho[..., None, None, None, None])
        * s2
        - 12.0 * muv[..., None, None, None, None]
        / (7.0 * coefficients.C1 * pressure[..., None, None, None, None])
        * s3
        - coefficients.C2 / (coefficients.C1 * rho[..., None, None, None, None]) * s4
    )

    # Eq. (25), psi_ijk, in the exact v3 source transcription.
    raw0_psi = np.einsum("...cab->...abc", gr) / rho[..., None, None, None]
    raw0_psi -= np.einsum("...ab,...c->...abc", rr, grho) / rho[..., None, None, None] ** 2
    p0 = stf3_project(raw0_psi)
    p1 = stf3_project(
        np.einsum("...ab,...c->...abc", rr + 7.0 * theta[..., None, None] * sigma, gtheta)
    )
    # q_i d_k u_j; STF3 makes j<->k transposition equivalent, but this is the
    # primary orientation for gu[derivative, component].
    p2 = stf3_project(np.einsum("...a,...cb->...abc", q, gu))
    p3 = stf3_project(np.einsum("...ab,...c->...abc", rr, div_sigma))
    # Primary Eq. (25): m_{mij} \partial_k u_m.  ``gu`` stores the
    # derivative index first, gu[k,m] = \partial_k u_m; the audited Fortran
    # oracle stores these two indices in the opposite order.  Keeping this
    # transpose explicit prevents a rotation-covariant but componentwise
    # wrong contraction from slipping through the tensor tests.
    eq25_m_grad_u = 54.0 / 7.0 * np.einsum("...eab,...ce->...abc", mm, gu)
    eq25_plus8_div_u = 8.0 * np.einsum("...abc,...->...abc", mm, div_u)
    eq25_minus6_div_u = -6.0 * np.einsum("...abc,...->...abc", mm, div_u)
    eq25_raw = eq25_m_grad_u + eq25_plus8_div_u + eq25_minus6_div_u
    p5 = stf3_project(eq25_raw)
    p6 = stf3_project(np.einsum("...ea,...bce->...abc", sigma, mm))
    p7 = stf3_project(np.einsum("...a,...bc->...abc", q, sigma))
    scalar = div_q + sigma_grad_u
    psi = (
        -27.0 * muv[..., None, None, None] / (7.0 * coefficients.Y1) * p0
        - 27.0 * muv[..., None, None, None]
        / (7.0 * coefficients.Y1 * pressure[..., None, None, None])
        * p1
        - 108.0 * muv[..., None, None, None]
        / (5.0 * coefficients.Y1 * rho[..., None, None, None])
        * p2
        + 27.0 * muv[..., None, None, None]
        / (7.0 * coefficients.Y1 * pressure[..., None, None, None] * rho[..., None, None, None])
        * p3
        + 6.0 * muv[..., None, None, None] * scalar[..., None, None, None]
        / (coefficients.Y1 * pressure[..., None, None, None] * rho[..., None, None, None])
        * mm
        - muv[..., None, None, None] / (coefficients.Y1 * rho[..., None, None, None]) * p5
        - coefficients.Y2 / (coefficients.Y1 * rho[..., None, None, None]) * p6
        - coefficients.Y3 / (coefficients.Y1 * rho[..., None, None, None]) * p7
    )

    # Eq. (26), Omega_i, including the v3 removal of the unsupported pressure
    # gradient addition that was present in stage-3 v2.
    div_R_over_rho = np.einsum("...bab->...a", gr) / rho[..., None]
    div_R_over_rho -= np.einsum("...ab,...b->...a", rr, grho) / rho[..., None] ** 2
    t1 = -4.0 * muv[..., None] * div_R_over_rho
    # q_j(d_j u_i + d_i u_j).  The two terms are deliberately written out;
    # unlike a single transpose they form the symmetric velocity gradient.
    t2_raw = np.einsum("...b,...ba->...a", q, gu) + np.einsum("...b,...ab->...a", q, gu)
    t2 = -56.0 * muv[..., None] / (5.0 * rho[..., None]) * t2_raw
    # m_ijk d_k u_j.  m is symmetric in j,k, so a transposed gu would be
    # numerically equivalent; retain the paper orientation explicitly.
    t3 = -8.0 * muv[..., None] / rho[..., None] * np.einsum("...abc,...cb->...a", mm, gu)
    t4_raw = np.einsum("...ab,...b->...a", 2.0 * theta[..., None, None] * sigma + rr, gtheta)
    t4 = -14.0 * muv[..., None] / pressure[..., None] * t4_raw
    t5 = 56.0 * muv[..., None] * q * scalar[..., None] / (3.0 * pressure[..., None] * rho[..., None])
    t6 = 4.0 * muv[..., None] / (pressure[..., None] * rho[..., None]) * np.einsum(
        "...ab,...b->...a", rr, div_sigma
    )
    t7 = (
        7.0
        * muv[..., None]
        * delta_moment[..., None]
        / (3.0 * pressure[..., None])
        * (div_sigma / rho[..., None] - 2.0 * gtheta)
    )
    t8_raw = 14.0 * np.einsum("...b,...ab->...a", q, sigma)
    t8_raw += 5.0 * np.einsum("...abc,...bc->...a", mm, sigma)
    t8 = -2.0 * t8_raw / (15.0 * rho[..., None])
    omega = (
        -7.0
        * muv[..., None]
        / 3.0
        * (gdelta / rho[..., None] - delta_moment[..., None] * grho / rho[..., None] ** 2)
        + t1
        + t2
        + t3
        + t4
        + t5
        + t6
        + t7
        + t8
    )

    if not np.isfinite(phi).all() or not np.isfinite(psi).all() or not np.isfinite(omega).all():
        raise FloatingPointError("R26 closure evaluation produced NaN or infinity")
    return R26Closures(
        phi=phi,
        psi=psi,
        Omega=omega,
        equation25_mode=equation25_mode,
        provenance=coefficients.provenance,
        coefficient_mode=coefficients.mode,
    )


def gu_emerson_closures(
    state: np.ndarray,
    *,
    mu: float | np.ndarray = 1.0,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    edge_order: int = 2,
    equation25_mode: str = "v3-literal",
    coefficient_mode: str = "jfm2009",
) -> R26Closures:
    """Convenience wrapper: differentiate a planar state grid and close R26."""

    u = validate_planar_state(state)
    gradients = finite_difference_gradients(u, x=x, y=y, dx=dx, dy=dy, edge_order=edge_order)
    return closures_from_tensors(
        planar_state_to_tensors(u),
        gradients,
        mu=mu,
        equation25_mode=equation25_mode,
        coefficient_mode=coefficient_mode,
    )


__all__ = [
    "CLOSURE_PROVENANCE",
    "PHI_C1",
    "PHI_C2",
    "PSI_Y1",
    "PSI_Y2",
    "PSI_Y3",
    "R26ClosureCoefficients",
    "closure_coefficients",
    "resolve_closure_mode",
    "R26Closures",
    "R26Gradients",
    "closures_from_tensors",
    "finite_difference_gradients",
    "gu_emerson_closures",
    "rotate_gradients",
    "stf2_project",
    "stf3_project",
    "stf4_project",
]
