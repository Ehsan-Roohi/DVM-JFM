#!/usr/bin/env python3
"""Pointwise steady nonlinear R26 bulk residuals of Gu--Emerson (2009).

This module implements the *raw* dimensional equations (2)--(4), (7)--(10),
and (16)--(21) in full three-dimensional tensor notation.  It is equally
usable with a consistently nondimensional state: no Knudsen-number convention
or reference scale is hard coded.  The scalar ``theta`` is ``R*T`` (specific
kinetic temperature), so ``p = rho*theta``.

The input gradients use the derivative-first convention of
``r26_tensor_closures.R26Gradients``.  Derivatives of the constitutive R26
closures are deliberately explicit point-kernel inputs: ``div(phi)``,
``div(psi)``, and ``grad(Omega)``.  A spatial discretization must compute those
from neighbouring closure values.  Consequently this file is independently
testable and does not hide a finite-difference stencil or a corner convention.

Scientific provenance and limitations
--------------------------------------
The equations and coefficients are those of Gu & Emerson, J. Fluid Mech. 636
(2009), DOI 10.1017/S002211200900768X, for the Maxwell-molecule nonlinear R26
model.  The Eq. (25) ambiguity is owned by ``r26_tensor_closures`` and remains
frozen in its audited ``v3-literal`` mode.  This is a bulk residual kernel,
not a wall treatment, nonlinear solver, cavity validation, or VHS-specific
collision model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from r26_state import StateTensors, planar_state_to_tensors, validate_planar_state
from r26_tensor_closures import (
    R26Closures,
    R26Gradients,
    closures_from_tensors,
    finite_difference_gradients,
    resolve_closure_mode,
    stf2_project,
    stf3_project,
)


BULK_PROVENANCE: Final[str] = (
    "Gu & Emerson, JFM 636 (2009), equations (2)-(4), (7)-(10), "
    "and (16)-(21); theta=R*T"
)


@dataclass(frozen=True)
class R26ClosureDerivatives:
    """Spatial derivatives of the high-order closures at one or more points.

    ``div_phi[..., i,j,k]`` is ``d_l phi[i,j,k,l]``;
    ``div_psi[..., i,j]`` is ``d_k psi[i,j,k]``; and
    ``grad_Omega[..., d,i]`` is ``d_d Omega[i]``.  Its trace is therefore
    ``div(Omega)``.
    """

    div_phi: np.ndarray
    div_psi: np.ndarray
    grad_Omega: np.ndarray

    @property
    def div_Omega(self) -> np.ndarray:
        """Return ``d_i Omega_i`` using the derivative-first convention."""

        return np.einsum("...ii->...", np.asarray(self.grad_Omega, dtype=float))


@dataclass(frozen=True)
class R26NonlinearSources:
    """Right-hand nonlinear sources in equations (8), (10), and (19)-(21)."""

    Sigma: np.ndarray
    Q: np.ndarray
    M: np.ndarray
    S: np.ndarray
    N: np.ndarray


@dataclass(frozen=True)
class R26BulkResidual:
    """Steady R26 residual, with every equation written as LHS minus RHS."""

    mass: np.ndarray
    momentum: np.ndarray
    theta: np.ndarray
    stress: np.ndarray
    heat_flux: np.ndarray
    m: np.ndarray
    R: np.ndarray
    Delta: np.ndarray
    provenance: str = BULK_PROVENANCE

    def as_planar17(self, *, atol: float = 2.0e-11) -> np.ndarray:
        """Pack a z-symmetric residual in the 17-state planar row ordering.

        Rows 9--16 are, in order, the three independent ``R`` balances, four
        independent ``m`` balances, and the ``Delta`` balance.  They are not
        R13 algebraic closure rows.
        """

        momentum = np.asarray(self.momentum, dtype=float)
        stress = np.asarray(self.stress, dtype=float)
        heat = np.asarray(self.heat_flux, dtype=float)
        rr = np.asarray(self.R, dtype=float)
        mm = np.asarray(self.m, dtype=float)
        odd = (
            momentum[..., 2],
            heat[..., 2],
            stress[..., 0, 2],
            stress[..., 1, 2],
            rr[..., 0, 2],
            rr[..., 1, 2],
            mm[..., 0, 0, 2],
            mm[..., 0, 1, 2],
            mm[..., 1, 1, 2],
            mm[..., 2, 2, 2],
        )
        if max((float(np.max(np.abs(a), initial=0.0)) for a in odd), default=0.0) > atol:
            raise ValueError("bulk residual violates planar z parity")
        values = (
            np.asarray(self.mass, dtype=float),
            momentum[..., 0],
            momentum[..., 1],
            np.asarray(self.theta, dtype=float),
            heat[..., 0],
            heat[..., 1],
            stress[..., 0, 0],
            stress[..., 0, 1],
            stress[..., 1, 1],
            rr[..., 0, 0],
            rr[..., 0, 1],
            rr[..., 1, 1],
            mm[..., 0, 0, 0],
            mm[..., 0, 0, 1],
            mm[..., 0, 1, 1],
            mm[..., 1, 1, 1],
            np.asarray(self.Delta, dtype=float),
        )
        return np.stack(np.broadcast_arrays(*values), axis=-1)


def _leading_shape(tensors: StateTensors) -> tuple[int, ...]:
    return np.asarray(tensors.rho, dtype=float).shape


def _check_tensor(
    array: np.ndarray,
    trailing: tuple[int, ...],
    leading: tuple[int, ...],
    name: str,
) -> np.ndarray:
    a = np.asarray(array, dtype=float)
    expected = leading + trailing
    if a.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {a.shape}")
    if not np.isfinite(a).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return a


def _broadcast_positive(value: float | np.ndarray, leading: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    try:
        a = np.broadcast_to(a, leading)
    except ValueError as exc:
        raise ValueError(f"{name} with shape {a.shape} cannot broadcast to {leading}") from exc
    if not np.isfinite(a).all() or np.any(a <= 0.0):
        raise FloatingPointError(f"{name} must be finite and positive")
    return a


def _validate_inputs(
    tensors: StateTensors,
    gradients: R26Gradients,
    closures: R26Closures,
    closure_derivatives: R26ClosureDerivatives,
    mu: float | np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray]:
    leading = _leading_shape(tensors)
    rho = _check_tensor(tensors.rho, (), leading, "rho")
    theta = _check_tensor(tensors.theta, (), leading, "theta")
    if np.any(rho <= 0.0) or np.any(theta <= 0.0):
        raise FloatingPointError("rho and theta must be positive")
    for name, shape in {
        "velocity": (3,),
        "heat_flux": (3,),
        "sigma": (3, 3),
        "R": (3, 3),
        "m": (3, 3, 3),
        "Delta": (),
    }.items():
        _check_tensor(getattr(tensors, name), shape, leading, name)
    for name, shape in {
        "rho": (3,),
        "velocity": (3, 3),
        "theta": (3,),
        "heat_flux": (3, 3),
        "sigma": (3, 3, 3),
        "R": (3, 3, 3),
        "m": (3, 3, 3, 3),
        "Delta": (3,),
    }.items():
        _check_tensor(getattr(gradients, name), shape, leading, f"gradient {name}")
    for name, shape in {"phi": (3, 3, 3, 3), "psi": (3, 3, 3), "Omega": (3,)}.items():
        _check_tensor(getattr(closures, name), shape, leading, f"closure {name}")
    if closures.equation25_mode != "v3-literal":
        raise ValueError("bulk kernel accepts only the audited 'v3-literal' Eq. (25) closure")
    for name, shape in {
        "div_phi": (3, 3, 3),
        "div_psi": (3, 3),
        "grad_Omega": (3, 3),
    }.items():
        _check_tensor(
            getattr(closure_derivatives, name), shape, leading, f"closure derivative {name}"
        )
    return leading, _broadcast_positive(mu, leading, "viscosity mu")


def _broadcast_acceleration(
    acceleration: np.ndarray | None, leading: tuple[int, ...]
) -> np.ndarray:
    if acceleration is None:
        return np.zeros(leading + (3,), dtype=float)
    a = np.asarray(acceleration, dtype=float)
    try:
        a = np.broadcast_to(a, leading + (3,))
    except ValueError as exc:
        raise ValueError(
            f"acceleration with shape {a.shape} cannot broadcast to {leading + (3,)}"
        ) from exc
    if not np.isfinite(a).all():
        raise FloatingPointError("acceleration contains NaN or infinity")
    return a


def gu_emerson_nonlinear_sources(
    tensors: StateTensors,
    gradients: R26Gradients,
    closures: R26Closures,
    closure_derivatives: R26ClosureDerivatives,
    *,
    mu: float | np.ndarray = 1.0,
) -> R26NonlinearSources:
    """Evaluate ``Sigma``, ``Q``, ``M``, ``S``, and ``N`` pointwise.

    This routine is intentionally public so every printed nonlinear source can
    be checked against an independent component-loop oracle without involving
    flux divergence or a PDE solver.
    """

    _, muv = _validate_inputs(tensors, gradients, closures, closure_derivatives, mu)
    rho = np.asarray(tensors.rho, dtype=float)
    velocity = np.asarray(tensors.velocity, dtype=float)
    del velocity  # The nonlinear sources depend on grad(u), not u itself.
    theta = np.asarray(tensors.theta, dtype=float)
    q = np.asarray(tensors.heat_flux, dtype=float)
    sigma = np.asarray(tensors.sigma, dtype=float)
    rr = np.asarray(tensors.R, dtype=float)
    mm = np.asarray(tensors.m, dtype=float)
    delta_moment = np.asarray(tensors.Delta, dtype=float)

    grho = np.asarray(gradients.rho, dtype=float)
    gu = np.asarray(gradients.velocity, dtype=float)
    gtheta = np.asarray(gradients.theta, dtype=float)
    gq = np.asarray(gradients.heat_flux, dtype=float)
    gs = np.asarray(gradients.sigma, dtype=float)
    gr = np.asarray(gradients.R, dtype=float)
    gm = np.asarray(gradients.m, dtype=float)
    phi = np.asarray(closures.phi, dtype=float)
    grad_omega = np.asarray(closure_derivatives.grad_Omega, dtype=float)

    pressure = rho * theta
    grad_pressure = theta[..., None] * grho + rho[..., None] * gtheta
    div_u = np.einsum("...ii->...", gu)
    div_q = np.einsum("...ii->...", gq)
    div_sigma = np.einsum("...jij->...i", gs)
    div_m = np.einsum("...kijk->...ij", gm)
    sigma_grad_u = np.einsum("...ij,...ij->...", sigma, gu)

    # Equation (8).
    sigma_source = (
        -4.0 / 5.0 * stf2_project(gq)
        - 2.0 * stf2_project(np.einsum("...ki,...kj->...ij", sigma, gu))
    )

    # Equation (10).
    q_deformation = (
        7.0 / 2.0 * np.einsum("...k,...ki->...i", q, gu)
        + np.einsum("...k,...ik->...i", q, gu)
        + q * div_u[..., None]
    )
    q_source = (
        -7.0 / 2.0 * np.einsum("...ik,...k->...i", sigma, gtheta)
        - theta[..., None] * div_sigma
        + np.einsum(
            "...ij,...j->...i",
            sigma / rho[..., None, None],
            grad_pressure + div_sigma,
        )
        - 2.0 / 5.0 * q_deformation
        # In Gu--Emerson Eq. (10), the Delta-gradient and m:grad(u) terms
        # are outside the preceding -2/5 parentheses.  Scaling them by
        # -2/5 (and reversing their sign) is a transcription error that is
        # invisible at equilibrium but directly corrupts heat flux and T.
        - np.asarray(gradients.Delta, dtype=float) / 6.0
        - np.einsum("...ijk,...kj->...i", mm, gu)
    )

    # Equation (19).
    grad_R_ij_k = np.moveaxis(gr, -3, -1)
    m_source = (
        3.0
        * stf3_project(
            np.einsum("...ij,...k->...ijk", sigma / rho[..., None, None], div_sigma)
        )
        - 12.0 / 5.0
        * stf3_project(np.einsum("...i,...kj->...ijk", q, gu))
        - 3.0 * stf3_project(np.einsum("...lij,...lk->...ijk", mm, gu))
        - 3.0 / 7.0 * stf3_project(grad_R_ij_k)
    )

    # Equation (20).
    s1 = (
        -2.0
        * pressure[..., None, None]
        / (3.0 * muv[..., None, None] * rho[..., None, None])
        * stf2_project(np.einsum("...ki,...jk->...ij", sigma, sigma))
    )
    s2 = -28.0 / 5.0 * stf2_project(np.einsum("...i,...j->...ij", q, gtheta))
    s3 = 28.0 / (5.0 * rho[..., None, None]) * stf2_project(
        np.einsum("...i,...j->...ij", q, div_sigma)
    )
    scalar = div_q + sigma_grad_u
    s4 = 14.0 / (3.0 * rho[..., None, None]) * sigma * scalar[..., None, None]
    sigma_deformation = np.einsum("...ki,...jk->...ij", sigma, gu)
    sigma_deformation += np.einsum("...ki,...kj->...ij", sigma, gu)
    s5 = -4.0 * theta[..., None, None] * stf2_project(sigma_deformation)
    s6 = 8.0 / 3.0 * theta[..., None, None] * sigma * div_u[..., None, None]
    s7 = -2.0 * theta[..., None, None] * div_m
    s8 = -9.0 * np.einsum("...ijk,...k->...ij", mm, gtheta)
    s9 = -2.0 * np.einsum("...ijkl,...lk->...ij", phi, gu)
    s10 = 2.0 / rho[..., None, None] * np.einsum(
        "...ijk,...k->...ij", mm, grad_pressure + div_sigma
    )
    r_deformation = 6.0 / 7.0 * rr * div_u[..., None, None]
    r_deformation += 4.0 / 5.0 * np.einsum("...ki,...jk->...ij", rr, gu)
    r_deformation += 2.0 * np.einsum("...ki,...kj->...ij", rr, gu)
    s11 = -stf2_project(r_deformation)
    s12 = -14.0 / 15.0 * delta_moment[..., None, None] * stf2_project(gu)
    s13 = -2.0 / 5.0 * stf2_project(grad_omega)
    r_source = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13

    # Equation (21).
    sigma_square = np.einsum("...ij,...ij->...", sigma, sigma)
    fourth_contraction = np.einsum(
        "...ij,...ji->...", 2.0 * theta[..., None, None] * sigma + rr, gu
    )
    delta_source = (
        -2.0 * pressure / (3.0 * muv * rho) * sigma_square
        - 4.0 * fourth_contraction
        + 8.0 / rho * np.einsum("...i,...i->...", q, div_sigma)
        - 20.0 * np.einsum("...i,...i->...", q, gtheta)
        - 4.0 / 3.0 * delta_moment * div_u
    )

    return R26NonlinearSources(
        Sigma=sigma_source,
        Q=q_source,
        M=m_source,
        S=r_source,
        N=delta_source,
    )


def steady_r26_bulk_residual(
    tensors: StateTensors,
    gradients: R26Gradients,
    closures: R26Closures,
    closure_derivatives: R26ClosureDerivatives,
    *,
    mu: float | np.ndarray = 1.0,
    acceleration: np.ndarray | None = None,
) -> R26BulkResidual:
    """Return the pointwise steady Gu--Emerson nonlinear R26 residual.

    Every returned field is ``LHS - RHS``.  Thus a converged solution has all
    fields zero.  ``acceleration`` is the body acceleration ``a_i`` from Eq.
    (3), not a force density.
    """

    leading, muv = _validate_inputs(tensors, gradients, closures, closure_derivatives, mu)
    body_acceleration = _broadcast_acceleration(acceleration, leading)
    sources = gu_emerson_nonlinear_sources(
        tensors, gradients, closures, closure_derivatives, mu=muv
    )

    rho = np.asarray(tensors.rho, dtype=float)
    velocity = np.asarray(tensors.velocity, dtype=float)
    theta = np.asarray(tensors.theta, dtype=float)
    q = np.asarray(tensors.heat_flux, dtype=float)
    sigma = np.asarray(tensors.sigma, dtype=float)
    rr = np.asarray(tensors.R, dtype=float)
    mm = np.asarray(tensors.m, dtype=float)
    delta_moment = np.asarray(tensors.Delta, dtype=float)

    grho = np.asarray(gradients.rho, dtype=float)
    gu = np.asarray(gradients.velocity, dtype=float)
    gtheta = np.asarray(gradients.theta, dtype=float)
    gq = np.asarray(gradients.heat_flux, dtype=float)
    gs = np.asarray(gradients.sigma, dtype=float)
    gr = np.asarray(gradients.R, dtype=float)
    gm = np.asarray(gradients.m, dtype=float)
    gdelta = np.asarray(gradients.Delta, dtype=float)

    pressure = rho * theta
    grad_pressure = theta[..., None] * grho + rho[..., None] * gtheta
    div_u = np.einsum("...ii->...", gu)
    div_q = np.einsum("...ii->...", gq)
    div_sigma = np.einsum("...jij->...i", gs)
    div_R = np.einsum("...jij->...i", gr)
    div_m = np.einsum("...kijk->...ij", gm)
    sigma_grad_u = np.einsum("...ij,...ij->...", sigma, gu)

    mass = np.einsum("...i,...i->...", velocity, grho) + rho * div_u
    momentum_flux = (
        np.einsum("...i,...j,...j->...i", velocity, velocity, grho)
        + rho[..., None] * np.einsum("...j,...ji->...i", velocity, gu)
        + rho[..., None] * velocity * div_u[..., None]
    )
    momentum = momentum_flux + div_sigma + grad_pressure - rho[..., None] * body_acceleration
    temperature_flux = theta * mass + rho * np.einsum("...i,...i->...", velocity, gtheta)
    temperature = temperature_flux + 2.0 / 3.0 * div_q
    temperature += 2.0 / 3.0 * (pressure * div_u + sigma_grad_u)

    conv_sigma = sigma * div_u[..., None, None]
    conv_sigma += np.einsum("...k,...kij->...ij", velocity, gs)
    stress = conv_sigma + div_m
    stress += pressure[..., None, None] / muv[..., None, None] * sigma
    stress += 2.0 * pressure[..., None, None] * stf2_project(gu)
    stress -= sources.Sigma

    conv_q = q * div_u[..., None] + np.einsum("...j,...ji->...i", velocity, gq)
    heat_flux = conv_q + 0.5 * div_R
    heat_flux += 2.0 * pressure[..., None] / (3.0 * muv[..., None]) * q
    heat_flux += 5.0 / 2.0 * pressure[..., None] * gtheta
    heat_flux -= sources.Q

    conv_m = mm * div_u[..., None, None, None]
    conv_m += np.einsum("...l,...lijk->...ijk", velocity, gm)
    grad_sigma_over_rho = np.moveaxis(gs, -3, -1) / rho[..., None, None, None]
    grad_sigma_over_rho -= np.einsum("...ij,...k->...ijk", sigma, grho) / rho[..., None, None, None] ** 2
    m_residual = conv_m + np.asarray(closure_derivatives.div_phi, dtype=float)
    m_residual += 3.0 * pressure[..., None, None, None] / (2.0 * muv[..., None, None, None]) * mm
    m_residual += 3.0 * pressure[..., None, None, None] * stf3_project(grad_sigma_over_rho)
    m_residual -= sources.M

    conv_R = rr * div_u[..., None, None] + np.einsum("...k,...kij->...ij", velocity, gr)
    grad_q_over_rho = np.swapaxes(gq, -2, -1) / rho[..., None, None]
    grad_q_over_rho -= np.einsum("...i,...j->...ij", q, grho) / rho[..., None, None] ** 2
    r_residual = conv_R + np.asarray(closure_derivatives.div_psi, dtype=float)
    r_residual += 7.0 * pressure[..., None, None] / (6.0 * muv[..., None, None]) * rr
    r_residual += 28.0 / 5.0 * pressure[..., None, None] * stf2_project(grad_q_over_rho)
    r_residual -= sources.S

    conv_delta = delta_moment * div_u + np.einsum("...i,...i->...", velocity, gdelta)
    div_q_over_rho = div_q / rho - np.einsum("...i,...i->...", q, grho) / rho**2
    delta_residual = conv_delta + closure_derivatives.div_Omega
    delta_residual += 2.0 * pressure / (3.0 * muv) * delta_moment
    delta_residual += 8.0 * pressure * div_q_over_rho
    delta_residual -= sources.N

    residual = R26BulkResidual(
        mass=mass,
        momentum=momentum,
        theta=temperature,
        stress=stress,
        heat_flux=heat_flux,
        m=m_residual,
        R=r_residual,
        Delta=delta_residual,
    )
    for name in ("mass", "momentum", "theta", "stress", "heat_flux", "m", "R", "Delta"):
        if not np.isfinite(np.asarray(getattr(residual, name))).all():
            raise FloatingPointError(f"R26 bulk residual {name} contains NaN or infinity")
    return residual


def _grid_derivative(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    edge_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a ``(ny,nx,...)`` field without moving tensor axes."""

    return (
        np.gradient(field, x, axis=1, edge_order=edge_order),
        np.gradient(field, y, axis=0, edge_order=edge_order),
    )


def closure_derivatives_on_grid(
    closures: R26Closures,
    x: np.ndarray,
    y: np.ndarray,
    *,
    edge_order: int = 2,
) -> R26ClosureDerivatives:
    """Differentiate closure arrays on a Cartesian 2D3V grid.

    The closure arrays must have leading shape ``(ny,nx)``.  ``x`` indexes
    axis 1, ``y`` indexes axis 0, and all z derivatives are exactly zero.
    The output retains the derivative-first ``grad_Omega[d,i]`` convention.
    """

    phi = np.asarray(closures.phi, dtype=float)
    psi = np.asarray(closures.psi, dtype=float)
    omega = np.asarray(closures.Omega, dtype=float)
    if phi.ndim != 6 or phi.shape[-4:] != (3, 3, 3, 3):
        raise ValueError(f"grid phi must have shape (ny,nx,3,3,3,3), got {phi.shape}")
    if psi.shape != phi.shape[:2] + (3, 3, 3):
        raise ValueError(f"grid psi has incompatible shape {psi.shape}")
    if omega.shape != phi.shape[:2] + (3,):
        raise ValueError(f"grid Omega has incompatible shape {omega.shape}")
    ny, nx = phi.shape[:2]
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.shape != (nx,) or not np.isfinite(xx).all() or np.any(np.diff(xx) <= 0.0):
        raise ValueError(f"x must be a strictly increasing vector of length {nx}")
    if yy.shape != (ny,) or not np.isfinite(yy).all() or np.any(np.diff(yy) <= 0.0):
        raise ValueError(f"y must be a strictly increasing vector of length {ny}")
    if edge_order not in (1, 2):
        raise ValueError("edge_order must be 1 or 2")
    if nx < edge_order + 1 or ny < edge_order + 1:
        raise ValueError(f"grid must have at least {edge_order + 1} nodes per direction")

    dphi_dx, dphi_dy = _grid_derivative(phi, xx, yy, edge_order=edge_order)
    dpsi_dx, dpsi_dy = _grid_derivative(psi, xx, yy, edge_order=edge_order)
    domega_dx, domega_dy = _grid_derivative(omega, xx, yy, edge_order=edge_order)
    div_phi = dphi_dx[..., 0] + dphi_dy[..., 1]
    div_psi = dpsi_dx[..., 0] + dpsi_dy[..., 1]
    grad_omega = np.stack((domega_dx, domega_dy, np.zeros_like(omega)), axis=-2)
    return R26ClosureDerivatives(
        div_phi=div_phi,
        div_psi=div_psi,
        grad_Omega=grad_omega,
    )


def _grid_force_density(
    body_force: np.ndarray | None,
    grid_shape: tuple[int, int],
) -> np.ndarray | None:
    """Broadcast a planar/full body-force density to ``(ny,nx,3)``."""

    if body_force is None:
        return None
    force = np.asarray(body_force, dtype=float)
    if force.shape[-1:] == (2,):
        full = np.zeros(force.shape[:-1] + (3,), dtype=float)
        full[..., :2] = force
        force = full
    try:
        force = np.broadcast_to(force, grid_shape + (3,))
    except ValueError as exc:
        raise ValueError(
            f"body_force with shape {force.shape} cannot broadcast to {grid_shape + (3,)}"
        ) from exc
    if not np.isfinite(force).all():
        raise FloatingPointError("body_force contains NaN or infinity")
    return force


def bulk_residual_grid(
    state: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float | np.ndarray = 1.0,
    body_force: np.ndarray | None = None,
    *,
    edge_order: int = 2,
    case: object | None = None,
    coefficient_mode: str | None = None,
) -> np.ndarray:
    """Evaluate the raw steady R26 bulk residual on a Cartesian grid.

    Parameters
    ----------
    state:
        ``(ny,nx,17)`` planar state in ``r26_state.STATE_ORDER``.
    x, y:
        Strictly increasing cell/node coordinates for axes 1 and 0.
    mu:
        Positive scalar or ``(ny,nx)`` dynamic viscosity in the same unit
        system as the state.
    body_force:
        Optional force *density* (not acceleration), with final length 2 or 3.
    case, coefficient_mode:
        Alternative complete closure-mode selectors.  A solver case propagates
        ``case.r26_closure_mode``; an explicit selector is useful for the
        standalone grid API.  Conflicting selectors are rejected.

    Notes
    -----
    This convenience API differentiates state and closure arrays with NumPy's
    Cartesian three-point stencil (or two-point stencil when
    ``edge_order=1``).  It is suitable for manufactured tests and for a solver
    interior.  A cavity solver must replace boundary rows with the R26 wall
    equations; these one-sided bulk residuals are not wall conditions.
    """

    u = validate_planar_state(state)
    if u.ndim != 3:
        raise ValueError(f"state grid must have shape (ny,nx,17), got {u.shape}")
    tensors = planar_state_to_tensors(u)
    gradients = finite_difference_gradients(u, x=x, y=y, edge_order=edge_order)
    closures = closures_from_tensors(
        tensors,
        gradients,
        mu=mu,
        coefficient_mode=resolve_closure_mode(
            case=case,
            coefficient_mode=coefficient_mode,
        ),
    )
    closure_derivatives = closure_derivatives_on_grid(
        closures, x, y, edge_order=edge_order
    )
    force_density = _grid_force_density(body_force, u.shape[:2])
    acceleration = None
    if force_density is not None:
        acceleration = force_density / np.asarray(tensors.rho)[..., None]
    return steady_r26_bulk_residual(
        tensors,
        gradients,
        closures,
        closure_derivatives,
        mu=mu,
        acceleration=acceleration,
    ).as_planar17()


def rotate_closures(closures: R26Closures, orthogonal: np.ndarray) -> R26Closures:
    """Transform closure values under a proper or improper orthogonal map."""

    q = np.asarray(orthogonal, dtype=float)
    if q.shape != (3, 3) or not np.allclose(q @ q.T, np.eye(3), rtol=0.0, atol=5.0e-13):
        raise ValueError("orthogonal must be a 3x3 orthogonal matrix")
    return R26Closures(
        phi=np.einsum("ai,bj,ck,dl,...ijkl->...abcd", q, q, q, q, closures.phi),
        psi=np.einsum("ai,bj,ck,...ijk->...abc", q, q, q, closures.psi),
        Omega=np.einsum("ai,...i->...a", q, closures.Omega),
        equation25_mode=closures.equation25_mode,
        provenance=closures.provenance,
        coefficient_mode=closures.coefficient_mode,
    )


def rotate_closure_derivatives(
    derivatives: R26ClosureDerivatives, orthogonal: np.ndarray
) -> R26ClosureDerivatives:
    """Transform closure divergences and ``grad(Omega)`` covariantly."""

    q = np.asarray(orthogonal, dtype=float)
    if q.shape != (3, 3) or not np.allclose(q @ q.T, np.eye(3), rtol=0.0, atol=5.0e-13):
        raise ValueError("orthogonal must be a 3x3 orthogonal matrix")
    return R26ClosureDerivatives(
        div_phi=np.einsum("ai,bj,ck,...ijk->...abc", q, q, q, derivatives.div_phi),
        div_psi=np.einsum("ai,bj,...ij->...ab", q, q, derivatives.div_psi),
        grad_Omega=np.einsum("ad,bi,...di->...ab", q, q, derivatives.grad_Omega),
    )


def rotate_bulk_residual(
    residual: R26BulkResidual, orthogonal: np.ndarray
) -> R26BulkResidual:
    """Transform a residual for a rotation/reflection covariance test."""

    q = np.asarray(orthogonal, dtype=float)
    if q.shape != (3, 3) or not np.allclose(q @ q.T, np.eye(3), rtol=0.0, atol=5.0e-13):
        raise ValueError("orthogonal must be a 3x3 orthogonal matrix")
    return R26BulkResidual(
        mass=np.asarray(residual.mass),
        momentum=np.einsum("ai,...i->...a", q, residual.momentum),
        theta=np.asarray(residual.theta),
        stress=np.einsum("ai,bj,...ij->...ab", q, q, residual.stress),
        heat_flux=np.einsum("ai,...i->...a", q, residual.heat_flux),
        m=np.einsum("ai,bj,ck,...ijk->...abc", q, q, q, residual.m),
        R=np.einsum("ai,bj,...ij->...ab", q, q, residual.R),
        Delta=np.asarray(residual.Delta),
        provenance=residual.provenance,
    )


__all__ = [
    "BULK_PROVENANCE",
    "R26BulkResidual",
    "R26ClosureDerivatives",
    "R26NonlinearSources",
    "bulk_residual_grid",
    "closure_derivatives_on_grid",
    "gu_emerson_nonlinear_sources",
    "rotate_bulk_residual",
    "rotate_closure_derivatives",
    "rotate_closures",
    "steady_r26_bulk_residual",
]
