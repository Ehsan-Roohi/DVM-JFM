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
    if f.shape == (nx, nv):
        return f
    if f.shape == (nv, nx):
        return f.T
    if f.shape[0] == nx:
        return f
    if f.shape[1] == nx:
        return f.T
    raise ValueError(f"Cannot orient f shape={f.shape}")


def load_grid(z):
    v = np.asarray(z[key(z, ["v", "vel", "velocity", "velocities"])])
    if v.ndim == 2 and v.shape[1] >= 3:
        vx, vy, vz = v[:,0], v[:,1], v[:,2]
    elif v.ndim == 2 and v.shape[0] >= 3:
        vx, vy, vz = v[0], v[1], v[2]
    else:
        raise ValueError(f"cannot parse v shape={v.shape}")
    w = np.asarray(z[key(z, ["w", "weights", "weight", "vw"])]).ravel()
    return vx.ravel().astype(float), vy.ravel().astype(float), vz.ravel().astype(float), w.astype(float)


def plateau_zero(y, frac=0.08):
    y = np.asarray(y, float).copy()
    n = len(y)
    ne = max(5, int(frac*n))
    idx = np.r_[np.arange(ne), np.arange(n-ne, n)]
    return y - np.mean(y[idx])


def moving_average(y, window):
    y = np.asarray(y, float)
    window = int(window)
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window//2
    yp = np.pad(y, (pad, pad), mode="edge")
    k = np.ones(window)/window
    return np.convolve(yp, k, mode="valid")


def rel_l2(a, b, mask=None):
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    den = np.linalg.norm(b[mask])
    if den < 1e-30:
        return np.nan
    return float(np.linalg.norm(a[mask]-b[mask])/den)


def corr(a, b, mask=None):
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    aa = a[mask]-np.mean(a[mask])
    bb = b[mask]-np.mean(b[mask])
    den = np.linalg.norm(aa)*np.linalg.norm(bb)
    if den < 1e-30:
        return np.nan
    return float(np.dot(aa, bb)/den)


def active_mask(y, thresh):
    peak = np.max(np.abs(y))
    if peak < 1e-30:
        return np.ones_like(y, dtype=bool)
    m = np.abs(y) >= thresh*peak
    if m.sum() < 10:
        return np.ones_like(y, dtype=bool)
    return m


def compute_moments(f, vx, vy, vz, w, rho, ux, T, chunk=64):
    nx = len(rho)
    q = np.zeros(nx)
    sig = np.zeros(nx)
    R = np.zeros(nx)
    Delta = np.zeros(nx)
    S = np.zeros(nx)
    L = np.zeros(nx)
    Hq = np.zeros(nx)
    Jd = np.zeros(nx)
    Kd = np.zeros(nx)

    for a in range(0, nx, chunk):
        b = min(nx, a+chunk)
        fb = f[a:b]
        rb = rho[a:b]
        ub = ux[a:b]
        Tb = T[a:b]

        cx = vx[None,:] - ub[:,None]
        cy = vy[None,:]
        cz = vz[None,:]
        c2 = cx*cx + cy*cy + cz*cz
        c4 = c2*c2

        M = rb[:,None]/((2*np.pi*Tb[:,None])**1.5)*np.exp(-0.5*c2/Tb[:,None])
        df = fb - M

        q[a:b] = np.sum(w[None,:]*0.5*c2*cx*fb, axis=1)
        sig[a:b] = np.sum(w[None,:]*(cx*cx-c2/3.0)*fb, axis=1)

        kr = c2*(cx*cx-c2/3.0)
        R[a:b] = np.sum(w[None,:]*kr*df, axis=1) - 7.0*Tb*sig[a:b]
        Delta[a:b] = np.sum(w[None,:]*c4*df, axis=1)
        S[a:b] = R[a:b] + Delta[a:b]/3.0

        L[a:b] = ub*q[a:b] + 0.5*(5.0*rb*Tb*Tb + 7.0*Tb*sig[a:b])

        # Heat-flux geometric term:
        # psi_q = 0.5 |c|^2 c_x, so
        # d psi_q / dx = -u_x' * (c_x^2 + 0.5 |c|^2).
        # Therefore dF_q/dx = Q_q + dpsi contribution,
        # and dS/dx = 2*(Q_q - dL/dx - u_x' Hq).
        Hq[a:b] = np.sum(
            w[None,:]*(vx[None,:])*(cx*cx + 0.5*c2)*fb,
            axis=1
        )

        Jd[a:b] = np.sum(w[None,:]*(vx[None,:])*c4*fb, axis=1)
        Kd[a:b] = np.sum(w[None,:]*(vx[None,:])*c2*cx*fb, axis=1)

    return q, sig, R, Delta, S, L, Hq, Jd, Kd


def integrate_source(x, dSdx):
    S = np.zeros_like(dSdx)
    S[1:] = np.cumsum(0.5*(dSdx[1:]+dSdx[:-1])*np.diff(x))
    return plateau_zero(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullstate", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--outdir", default="figures/week2_jfm_upgrade_shakhov")
    ap.add_argument("--Pr", type=float, default=2.0/3.0)
    ap.add_argument("--Kn", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--smooth-deriv", type=int, default=7)
    ap.add_argument("--active-thresh", type=float, default=0.02)
    args = ap.parse_args()

    outdir = Path(args.outdir)/args.case
    outdir.mkdir(parents=True, exist_ok=True)
    outstem = outdir/args.case

    z = np.load(args.fullstate, allow_pickle=True)
    x = np.asarray(z[key(z, ["x", "x_mfp"])]).astype(float)
    rho = np.asarray(z[key(z, ["rho"])]).astype(float)
    ux = np.asarray(z[key(z, ["ux", "u", "u_x"])]).astype(float)
    T = np.asarray(z[key(z, ["T"])]).astype(float)

    vx, vy, vz, w = load_grid(z)
    f = orient_f(z[key(z, ["f", "F", "dist", "distribution"])], len(x), len(w)).astype(float)

    q, sig, R, Delta, S, L, Hq, Jd, Kd = compute_moments(
        f, vx, vy, vz, w, rho, ux, T, chunk=args.chunk
    )

    R = plateau_zero(R)
    Delta = plateau_zero(Delta)
    S = plateau_zero(S)

    # Shakhov heat-flux budget:
    # F_q = L + 0.5 S, source = -Pr*q/Kn.
    # Because psi_q = 0.5 |c|^2 c_x depends on u_x(x),
    # the moving peculiar-velocity frame contributes
    # -u_x' Hq to dF_q/dx.
    Ls = moving_average(L, args.smooth_deriv)
    Hqs = moving_average(Hq, args.smooth_deriv)
    uxs_q = moving_average(ux, args.smooth_deriv)

    dLdx = np.gradient(Ls, x)
    dudx_q = np.gradient(uxs_q, x)
    heat_geom = dudx_q * Hqs

    Qq = -args.Pr*q/args.Kn
    dSdx = 2.0*(Qq - heat_geom - dLdx)
    S_rec = integrate_source(x, dSdx)

    # Delta budget unchanged under Shakhov correction.
    Jds = moving_average(Jd, args.smooth_deriv)
    uxs = moving_average(ux, args.smooth_deriv)
    Kds = moving_average(Kd, args.smooth_deriv)
    dJdx = np.gradient(Jds, x)
    dudx = np.gradient(uxs, x)
    forcing_delta = dJdx + 4.0*dudx*Kds
    Delta_rec = plateau_zero(-args.Kn*forcing_delta)

    R_no_delta = S_rec.copy()
    R_two_budget = S_rec - Delta_rec/3.0
    R_full_delta = S_rec - Delta/3.0

    mask = active_mask(S, args.active_thresh)

    metrics = {
        "case": args.case,
        "Pr": args.Pr,
        "Kn": args.Kn,
        "smooth_deriv": args.smooth_deriv,
        "S_rec_relL2_active": rel_l2(S_rec, S, mask),
        "S_rec_corr_active": corr(S_rec, S, mask),
        "Delta_rec_relL2_active": rel_l2(Delta_rec, Delta, mask),
        "Delta_rec_corr_active": corr(Delta_rec, Delta, mask),
        "R_delta0_relL2_active": rel_l2(R_no_delta, R, mask),
        "R_fullDelta_relL2_active": rel_l2(R_full_delta, R, mask),
        "R_twoBudget_relL2_active": rel_l2(R_two_budget, R, mask),
        "R_twoBudget_corr_active": corr(R_two_budget, R, mask),
    }

    pd.DataFrame([metrics]).to_csv(outstem.with_suffix(".metrics.csv"), index=False)

    pd.DataFrame({
        "x": x,
        "rho": rho,
        "ux": ux,
        "T": T,
        "qx": q,
        "sigma_xx": sig,
        "R": R,
        "Delta": Delta,
        "S": S,
        "S_rec_shakhov": S_rec,
        "Qq_shakhov": Qq,
        "dLdx_heatflux": dLdx,
        "heatflux_geom": heat_geom,
        "Delta_rec_budget": Delta_rec,
        "R_delta0": R_no_delta,
        "R_fullDelta": R_full_delta,
        "R_twoBudget": R_two_budget,
        "active_mask": mask.astype(int),
    }).to_csv(outstem.with_suffix(".profiles.csv"), index=False)

    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)

    axs[0].plot(x, S, "k-", label="DVM S")
    axs[0].plot(x, S_rec, "r--", label="Shakhov S-budget")
    axs[0].set_title(r"Heat-flux budget, \(Q_q=-Pr\,q/Kn\)")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(fontsize=8)

    axs[1].plot(x, Delta, "k-", label=r"DVM $\Delta$")
    axs[1].plot(x, Delta_rec, "r--", label=r"$\Delta$-budget")
    axs[1].set_title(r"Scalar-excess budget")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(fontsize=8)

    axs[2].plot(x, R, "k-", label=r"DVM $R^{cl}_{xx}$")
    axs[2].plot(x, R_no_delta, "r--", label=r"$S$, $\Delta=0$")
    axs[2].plot(x, R_two_budget, "C2--", label="two-budget")
    axs[2].set_title(r"Two-budget \(R^{cl}_{xx}\) recovery")
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(fontsize=8)

    for ax in axs:
        ax.set_xlabel(r"$(x-x_s)/\lambda_1$")

    fig.suptitle(args.case + rf": Shakhov spot-check, \(Pr={args.Pr:.3g}\)", fontsize=12)
    fig.savefig(str(outstem)+".pdf", bbox_inches="tight")
    fig.savefig(str(outstem)+".png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    print("\n=== metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:30s}: {v:.6g}")
        else:
            print(f"{k:30s}: {v}")

    print("\n[saved]")
    print(outstem.with_suffix(".metrics.csv"))
    print(outstem.with_suffix(".profiles.csv"))
    print(str(outstem)+".pdf")
    print(str(outstem)+".png")


if __name__ == "__main__":
    main()
