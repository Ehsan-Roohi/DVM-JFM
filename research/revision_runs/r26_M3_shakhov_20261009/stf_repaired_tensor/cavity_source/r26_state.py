#!/usr/bin/env python3
"""Planar 2D3V state and full three-dimensional STF tensor mappings for R26.

The nonlinear R26 equations have 26 independent moments in a general 3-D
flow.  A z-homogeneous, z-reflection-symmetric cavity retains 17 independent
unknowns.  This module makes that reduction explicit; no high-order moment is
silently discarded or re-labelled as an R13 closure variable.

The ASTR packing used here is the one in ``methodmoment.F90``:

* STF2: ``xx, xy, xz, yy, yz``;
* STF3: ``xxx, xxy, xxz, xyy, yyy, yyz, xyz``;
* STF4: ``xxxx, xxxy, xxxz, xxyy, xxyz, xyyy, xyyz, yyyy, yyyz``.

All routines accept arbitrary leading dimensions and put tensor indices last.
This is private verification code and is intentionally independent of the
public repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Final

import numpy as np


STATE_ORDER: Final[tuple[str, ...]] = (
    "rho",
    "vx",
    "vy",
    "theta",
    "qx",
    "qy",
    "sigma_xx",
    "sigma_xy",
    "sigma_yy",
    "R_xx",
    "R_xy",
    "R_yy",
    "m_xxx",
    "m_xxy",
    "m_xyy",
    "m_yyy",
    "Delta",
)
NVAR: Final[int] = len(STATE_ORDER)
STATE_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(STATE_ORDER)}


@dataclass(frozen=True)
class StateTensors:
    """A full 3-D tensor view of one or more R26 states."""

    rho: np.ndarray
    velocity: np.ndarray
    theta: np.ndarray
    heat_flux: np.ndarray
    sigma: np.ndarray
    R: np.ndarray
    m: np.ndarray
    Delta: np.ndarray


@dataclass(frozen=True)
class ASTRPackedState:
    """State components in the legacy ASTR full-3D component ordering."""

    rho: np.ndarray
    velocity: np.ndarray
    theta: np.ndarray
    heat_flux: np.ndarray
    sigma5: np.ndarray
    R5: np.ndarray
    m7: np.ndarray
    Delta: np.ndarray


def validate_planar_state(state: np.ndarray, *, physical: bool = True) -> np.ndarray:
    """Return a float state after shape, finiteness, and positivity checks."""

    u = np.asarray(state, dtype=float)
    if u.ndim < 1 or u.shape[-1] != NVAR:
        raise ValueError(f"state must end in {NVAR} components, got {u.shape}")
    if not np.isfinite(u).all():
        raise FloatingPointError("R26 state contains NaN or infinity")
    if physical and (np.any(u[..., 0] <= 0.0) or np.any(u[..., 3] <= 0.0)):
        raise FloatingPointError("R26 state requires positive rho and theta")
    return u


def _check_last_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(array, dtype=float)
    if a.shape[-len(shape) :] != shape:
        raise ValueError(f"{name} must end in {shape}, got {a.shape}")
    return a


def pack_stf2(tensor: np.ndarray) -> np.ndarray:
    """Pack a symmetric trace-free rank-2 tensor in ASTR order."""

    a = _check_last_shape(tensor, (3, 3), "STF2 tensor")
    return np.stack((a[..., 0, 0], a[..., 0, 1], a[..., 0, 2], a[..., 1, 1], a[..., 1, 2]), axis=-1)


def unpack_stf2(packed: np.ndarray) -> np.ndarray:
    """Unpack ASTR's five independent STF2 components into a 3x3 tensor."""

    v = _check_last_shape(packed, (5,), "packed STF2")
    a = np.zeros(v.shape[:-1] + (3, 3), dtype=float)
    a[..., 0, 0] = v[..., 0]
    a[..., 0, 1] = a[..., 1, 0] = v[..., 1]
    a[..., 0, 2] = a[..., 2, 0] = v[..., 2]
    a[..., 1, 1] = v[..., 3]
    a[..., 1, 2] = a[..., 2, 1] = v[..., 4]
    a[..., 2, 2] = -v[..., 0] - v[..., 3]
    return a


def pack_stf3(tensor: np.ndarray) -> np.ndarray:
    """Pack a symmetric trace-free rank-3 tensor in ASTR order."""

    a = _check_last_shape(tensor, (3, 3, 3), "STF3 tensor")
    return np.stack(
        (
            a[..., 0, 0, 0],
            a[..., 0, 0, 1],
            a[..., 0, 0, 2],
            a[..., 0, 1, 1],
            a[..., 1, 1, 1],
            a[..., 1, 1, 2],
            a[..., 0, 1, 2],
        ),
        axis=-1,
    )


def _assign_symmetric(out: np.ndarray, indices: tuple[int, ...], value: np.ndarray) -> None:
    for component in set(permutations(indices)):
        out[(Ellipsis,) + component] = value


def unpack_stf3(packed: np.ndarray) -> np.ndarray:
    """Unpack ASTR's seven independent STF3 components into a 3x3x3 tensor."""

    v = _check_last_shape(packed, (7,), "packed STF3")
    a = np.zeros(v.shape[:-1] + (3, 3, 3), dtype=float)
    independent = (
        ((0, 0, 0), v[..., 0]),
        ((0, 0, 1), v[..., 1]),
        ((0, 0, 2), v[..., 2]),
        ((0, 1, 1), v[..., 3]),
        ((1, 1, 1), v[..., 4]),
        ((1, 1, 2), v[..., 5]),
        ((0, 1, 2), v[..., 6]),
        ((0, 2, 2), -v[..., 0] - v[..., 3]),
        ((1, 2, 2), -v[..., 1] - v[..., 4]),
        ((2, 2, 2), -v[..., 2] - v[..., 5]),
    )
    for indices, value in independent:
        _assign_symmetric(a, indices, value)
    return a


def pack_stf4(tensor: np.ndarray) -> np.ndarray:
    """Pack a symmetric trace-free rank-4 tensor in ASTR order."""

    a = _check_last_shape(tensor, (3, 3, 3, 3), "STF4 tensor")
    return np.stack(
        (
            a[..., 0, 0, 0, 0],
            a[..., 0, 0, 0, 1],
            a[..., 0, 0, 0, 2],
            a[..., 0, 0, 1, 1],
            a[..., 0, 0, 1, 2],
            a[..., 0, 1, 1, 1],
            a[..., 0, 1, 1, 2],
            a[..., 1, 1, 1, 1],
            a[..., 1, 1, 1, 2],
        ),
        axis=-1,
    )


def unpack_stf4(packed: np.ndarray) -> np.ndarray:
    """Unpack ASTR's nine independent STF4 components into a 3^4 tensor."""

    v = _check_last_shape(packed, (9,), "packed STF4")
    a = np.zeros(v.shape[:-1] + (3, 3, 3, 3), dtype=float)
    independent = (
        ((0, 0, 0, 0), v[..., 0]),
        ((0, 0, 0, 1), v[..., 1]),
        ((0, 0, 0, 2), v[..., 2]),
        ((0, 0, 1, 1), v[..., 3]),
        ((0, 0, 1, 2), v[..., 4]),
        ((0, 1, 1, 1), v[..., 5]),
        ((0, 1, 1, 2), v[..., 6]),
        ((1, 1, 1, 1), v[..., 7]),
        ((1, 1, 1, 2), v[..., 8]),
        ((0, 0, 2, 2), -v[..., 0] - v[..., 3]),
        ((0, 1, 2, 2), -v[..., 1] - v[..., 5]),
        ((0, 2, 2, 2), -v[..., 2] - v[..., 6]),
        ((1, 1, 2, 2), -v[..., 3] - v[..., 7]),
        ((1, 2, 2, 2), -v[..., 4] - v[..., 8]),
        ((2, 2, 2, 2), v[..., 0] + 2.0 * v[..., 3] + v[..., 7]),
    )
    for indices, value in independent:
        _assign_symmetric(a, indices, value)
    return a


def planar_state_to_tensors(state: np.ndarray) -> StateTensors:
    """Expand the 17-component z-symmetric state into full 3-D tensors."""

    u = validate_planar_state(state)
    leading = u.shape[:-1]
    velocity = np.zeros(leading + (3,), dtype=float)
    velocity[..., :2] = u[..., 1:3]
    heat_flux = np.zeros_like(velocity)
    heat_flux[..., :2] = u[..., 4:6]

    sigma5 = np.stack(
        (u[..., 6], u[..., 7], np.zeros(leading), u[..., 8], np.zeros(leading)), axis=-1
    )
    R5 = np.stack(
        (u[..., 9], u[..., 10], np.zeros(leading), u[..., 11], np.zeros(leading)), axis=-1
    )
    m7 = np.stack(
        (
            u[..., 12],
            u[..., 13],
            np.zeros(leading),
            u[..., 14],
            u[..., 15],
            np.zeros(leading),
            np.zeros(leading),
        ),
        axis=-1,
    )
    return StateTensors(
        rho=u[..., 0],
        velocity=velocity,
        theta=u[..., 3],
        heat_flux=heat_flux,
        sigma=unpack_stf2(sigma5),
        R=unpack_stf2(R5),
        m=unpack_stf3(m7),
        Delta=u[..., 16],
    )


def tensors_to_planar_state(tensors: StateTensors, *, atol: float = 1.0e-12) -> np.ndarray:
    """Reduce full tensors to 17 components after enforcing planar parity/STF."""

    rho = np.asarray(tensors.rho, dtype=float)
    theta = np.asarray(tensors.theta, dtype=float)
    velocity = _check_last_shape(tensors.velocity, (3,), "velocity")
    heat_flux = _check_last_shape(tensors.heat_flux, (3,), "heat flux")
    sigma = _check_last_shape(tensors.sigma, (3, 3), "sigma")
    R = _check_last_shape(tensors.R, (3, 3), "R")
    m = _check_last_shape(tensors.m, (3, 3, 3), "m")

    if np.max(np.abs(velocity[..., 2]), initial=0.0) > atol:
        raise ValueError("velocity violates planar z parity")
    if np.max(np.abs(heat_flux[..., 2]), initial=0.0) > atol:
        raise ValueError("heat flux violates planar z parity")
    if np.max(np.abs(sigma[..., :2, 2]), initial=0.0) > atol or np.max(np.abs(R[..., :2, 2]), initial=0.0) > atol:
        raise ValueError("rank-2 tensor violates planar z parity")
    odd_z = [m[..., 0, 0, 2], m[..., 0, 1, 2], m[..., 1, 1, 2], m[..., 2, 2, 2]]
    if max((float(np.max(np.abs(a), initial=0.0)) for a in odd_z), default=0.0) > atol:
        raise ValueError("rank-3 tensor violates planar z parity")

    if np.max(np.abs(np.trace(sigma, axis1=-2, axis2=-1)), initial=0.0) > atol:
        raise ValueError("sigma is not trace free")
    if np.max(np.abs(np.trace(R, axis1=-2, axis2=-1)), initial=0.0) > atol:
        raise ValueError("R is not trace free")
    if np.max(np.abs(np.einsum("...iik->...k", m)), initial=0.0) > atol:
        raise ValueError("m is not trace free")

    values = (
        rho,
        velocity[..., 0],
        velocity[..., 1],
        theta,
        heat_flux[..., 0],
        heat_flux[..., 1],
        sigma[..., 0, 0],
        sigma[..., 0, 1],
        sigma[..., 1, 1],
        R[..., 0, 0],
        R[..., 0, 1],
        R[..., 1, 1],
        m[..., 0, 0, 0],
        m[..., 0, 0, 1],
        m[..., 0, 1, 1],
        m[..., 1, 1, 1],
        np.asarray(tensors.Delta, dtype=float),
    )
    state = np.stack(np.broadcast_arrays(*values), axis=-1)
    return validate_planar_state(state)


def astr_pack_planar_state(state: np.ndarray) -> ASTRPackedState:
    """Expose the exact 2D3V-to-ASTR component mapping used by the port."""

    tensors = planar_state_to_tensors(state)
    return ASTRPackedState(
        rho=tensors.rho,
        velocity=tensors.velocity,
        theta=tensors.theta,
        heat_flux=tensors.heat_flux,
        sigma5=pack_stf2(tensors.sigma),
        R5=pack_stf2(tensors.R),
        m7=pack_stf3(tensors.m),
        Delta=tensors.Delta,
    )


def rotate_tensors(tensors: StateTensors, orthogonal: np.ndarray) -> StateTensors:
    """Apply a proper or improper orthogonal transform to all physical tensors."""

    q = np.asarray(orthogonal, dtype=float)
    if q.shape != (3, 3) or not np.allclose(q @ q.T, np.eye(3), rtol=0.0, atol=5.0e-13):
        raise ValueError("orthogonal must be a 3x3 orthogonal matrix")
    return StateTensors(
        rho=np.asarray(tensors.rho),
        velocity=np.einsum("ai,...i->...a", q, tensors.velocity),
        theta=np.asarray(tensors.theta),
        heat_flux=np.einsum("ai,...i->...a", q, tensors.heat_flux),
        sigma=np.einsum("ai,bj,...ij->...ab", q, q, tensors.sigma),
        R=np.einsum("ai,bj,...ij->...ab", q, q, tensors.R),
        m=np.einsum("ai,bj,ck,...ijk->...abc", q, q, q, tensors.m),
        Delta=np.asarray(tensors.Delta),
    )


__all__ = [
    "ASTRPackedState",
    "NVAR",
    "STATE_INDEX",
    "STATE_ORDER",
    "StateTensors",
    "astr_pack_planar_state",
    "pack_stf2",
    "pack_stf3",
    "pack_stf4",
    "planar_state_to_tensors",
    "rotate_tensors",
    "tensors_to_planar_state",
    "unpack_stf2",
    "unpack_stf3",
    "unpack_stf4",
    "validate_planar_state",
]
