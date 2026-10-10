#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def key(z, names, required=True):
    for n in names:
        if n in z.files:
            return n
    if required:
        raise KeyError(f"Missing {names}; available={z.files}")
    return None


def orient_f(f, nx, nv):
    f = np.asarray(f)
    if f.ndim != 2:
        raise ValueError(f"f must be 2D, got {f.shape}")
    if f.shape == (nx, nv):
        return f
    if f.shape == (nv, nx):
        return f.T
    if f.shape[0] == nx:
        return f
    if f.shape[1] == nx:
        return f.T
    raise ValueError(f"Cannot orient f with shape {f.shape}, nx={nx}, nv={nv}")


def load_velocity_grid(z):
    # Preferred: combined velocity array.
    vk = key(z, ["v", "vel", "velocity", "velocities"], required=False)
    if vk is not None:
        v = np.asarray(z[vk])
        if v.ndim == 2 and v.shape[1] >= 3:
            vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        elif v.ndim == 2 and v.shape[0] >= 3:
            vx, vy, vz = v[0], v[1], v[2]
        else:
            raise ValueError(f"Cannot parse velocity array {vk}, shape={v.shape}")
    else:
        vx = np.asarray(z[key(z, ["vx", "v_x", "Vx"])])
        vy = np.asarray(z[key(z, ["vy", "v_y", "Vy"])])
        vz = np.asarray(z[key(z, ["vz", "v_z", "Vz"])])

    wk = key(z, ["w", "weights", "weight", "vw"])
    w = np.asarray(z[wk])

    vx = vx.ravel().astype(float)
    vy = vy.ravel().astype(float)
    vz = vz.ravel().astype(float)
    w = w.ravel().astype(float)

    if not (len(vx) == len(vy) == len(vz) == len(w)):
        raise ValueError(
            f"Velocity-grid mismatch: {len(vx)}, {len(vy)}, {len(vz)}, {len(w)}"
        )

    return vx, vy, vz, w


def moving_average(y, window):
    y = np.asarray(y, float)
    window = int(window)
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    k = np.ones(window) / window
    return np.convolve(yp, k, mode="valid")


def plateau_zero(y, frac=0.08):
    y = np.asarray(y, float).copy()
    n = len(y)
    ne = max(5, int(frac*n))
    idx = np.r_[np.arange(ne), np.arange(n-ne, n)]
    return y - np.mean(y[idx])


def rel_l2(a, b, mask=None):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    den = np.linalg.norm(b[mask])
    if den < 1e-30:
        return np.nan
    return float(np.linalg.norm(a[mask]-b[mask])/den)


def corr(a, b, mask=None):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    aa = a[mask] - np.mean(a[mask])
    bb = b[mask] - np.mean(b[mask])
    den = np.linalg.norm(aa)*np.linalg.norm(bb)
    if den < 1e-30:
        return np.nan
    return float(np.dot(aa, bb)/den)


def active_mask(y, thresh):
    y = np.asarray(y, float)
    peak = np.max(np.abs(y))
    if peak < 1e-30:
        return np.ones_like(y, dtype=bool)
    m = np.abs(y) >= thresh*peak
    if m.sum() < 10:
        return np.ones_like(y, dtype=bool)
    return m


def process_chunk(f, vx, vy, vz, w, rho, ux, T):
    cx = vx[None, :] - ux[:, None]
    cy = vy[None, :]
    cz = vz[None, :]
    vxx = vx[None, :]

    c2 = cx*cx + cy*cy + cz*cz
    c4 = c2*c2

    # Direct central fourth moment and Delta.
    C4 = np.sum(w[None, :] * c4 * f, axis=1)
    Delta = C4 - 15.0*rho*T*T

    # Delta flux: J = ∫ v_x |c|^4 f dv.
    J = np.sum(w[None, :] * vxx * c4 * f, axis=1)

    # Geometric term helper: K = ∫ v_x |c|^2 c_x f dv.
    K = np.sum(w[None, :] * vxx * c2 * cx * f, axis=1)

    return C4, Delta, J, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullstate", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--outdir", default="figures/week2_jfm_upgrade_delta_budget")
    ap.add_argument("--kn", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--smooth-deriv", type=int, default=1)
    ap.add_argument("--active-thresh", type=float, default=0.02)
    args = ap.parse_args()

    outdir = Path(args.outdir) / args.case
    outdir.mkdir(parents=True, exist_ok=True)
    outstem = outdir / args.case

    z = np.load(args.fullstate, allow_pickle=True)

    x = np.asarray(z[key(z, ["x", "x_mfp"])]).astype(float)
    rho = np.asarray(z[key(z, ["rho", "rho_file"])]).astype(float)
    ux = np.asarray(z[key(z, ["ux", "u", "u_x"])]).astype(float)
    T = np.asarray(z[key(z, ["T", "temp"])]).astype(float)

    vx, vy, vz, w = load_velocity_grid(z)
    nx = len(x)
    nv = len(w)

    fk = key(z, ["f", "F", "dist", "distribution"])
    f = orient_f(z[fk], nx, nv).astype(float)

    print("="*72)
    print(f"[case] {args.case}")
    print(f"[fullstate] {args.fullstate}")
    print(f"[nx,nv] {nx}, {nv}")
    print(f"[chunk] {args.chunk}")
    print(f"[smooth_deriv] {args.smooth_deriv}")
    print("="*72, flush=True)

    C4 = np.zeros(nx)
    Delta = np.zeros(nx)
    J = np.zeros(nx)
    K = np.zeros(nx)

    for a in range(0, nx, args.chunk):
        b = min(nx, a+args.chunk)
        C4[a:b], Delta[a:b], J[a:b], K[a:b] = process_chunk(
            f[a:b], vx, vy, vz, w, rho[a:b], ux[a:b], T[a:b]
        )
        if a == 0 or b == nx or (a//args.chunk) % 20 == 0:
            print(f"[chunk] {a}:{b}", flush=True)

    # Plateau-zero the diagnostic Delta for comparison.
    Delta = plateau_zero(Delta)

    # Derivative terms.
    J_s = moving_average(J, args.smooth_deriv)
    ux_s = moving_average(ux, args.smooth_deriv)
    K_s = moving_average(K, args.smooth_deriv)

    dJdx = np.gradient(J_s, x)
    dudx = np.gradient(ux_s, x)

    geom = 4.0 * dudx * K_s

    # Central c^4 budget:
    # Q_delta = dJ/dx + 4 u_x' K
    # BGK source is Q_delta = -Delta/Kn.
    forcing = dJdx + geom
    Delta_rec = -args.kn * forcing
    Delta_rec = plateau_zero(Delta_rec)

    residual = forcing + Delta/args.kn

    mask = active_mask(Delta, args.active_thresh)

    metrics = {
        "case": args.case,
        "fullstate": args.fullstate,
        "kn": args.kn,
        "smooth_deriv": args.smooth_deriv,
        "active_thresh": args.active_thresh,
        "Delta_rec_relL2_full": rel_l2(Delta_rec, Delta),
        "Delta_rec_relL2_active": rel_l2(Delta_rec, Delta, mask),
        "Delta_rec_corr_active": corr(Delta_rec, Delta, mask),
        "residual_rel_active": float(
            np.linalg.norm(residual[mask]) /
            max(np.linalg.norm((Delta/args.kn)[mask]), 1e-300)
        ),
        "Delta_peak": float(np.max(np.abs(Delta))),
        "Delta_rec_peak": float(np.max(np.abs(Delta_rec))),
    }

    pd.DataFrame([metrics]).to_csv(outstem.with_suffix(".metrics.csv"), index=False)

    df = pd.DataFrame({
        "x": x,
        "rho": rho,
        "ux": ux,
        "T": T,
        "C4": C4,
        "Delta": Delta,
        "J_Delta": J,
        "K_geom": K,
        "dJdx": dJdx,
        "dudx": dudx,
        "geom": geom,
        "forcing": forcing,
        "Delta_rec_from_budget": Delta_rec,
        "budget_residual": residual,
        "active_mask": mask.astype(int),
    })
    df.to_csv(outstem.with_suffix(".profiles.csv"), index=False)

    # Plot.
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)

    axs[0].plot(x, Delta, "k-", lw=1.8, label=r"DVM $\Delta$")
    axs[0].plot(x, Delta_rec, "r--", lw=1.5, label=r"budget rec.")
    axs[0].set_title(r"Scalar-excess budget")
    axs[0].set_xlabel(r"$(x-x_s)/\lambda_1$")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(fontsize=8, frameon=True)

    axs[1].plot(x, dJdx, "C0-", label=r"$\partial_x J_\Delta$")
    axs[1].plot(x, geom, "C1-", label=r"geometric")
    axs[1].plot(x, forcing, "k--", label=r"sum")
    axs[1].set_title(r"Budget terms")
    axs[1].set_xlabel(r"$(x-x_s)/\lambda_1$")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(fontsize=8, frameon=True)

    axs[2].plot(x, residual, "C3-")
    axs[2].set_title(r"Residual: forcing $+\Delta/Kn$")
    axs[2].set_xlabel(r"$(x-x_s)/\lambda_1$")
    axs[2].grid(True, alpha=0.25)

    fig.suptitle(args.case + r": exact scalar-excess budget from full DVM state", fontsize=12)

    fig.savefig(str(outstem)+".pdf", bbox_inches="tight")
    fig.savefig(str(outstem)+".png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    print("\n=== metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:28s}: {v:.6g}")
        else:
            print(f"{k:28s}: {v}")

    print("\n[saved]")
    print(outstem.with_suffix(".metrics.csv"))
    print(outstem.with_suffix(".profiles.csv"))
    print(str(outstem)+".pdf")
    print(str(outstem)+".png")


if __name__ == "__main__":
    main()
