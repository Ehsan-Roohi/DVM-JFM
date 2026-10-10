#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src/week2_jfm_upgrade")

from run_shakhov_conservative_sweep import (
    key, orient_f, load_grid, moments, build_target,
    high_moments_chunked
)


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
    if den < 1e-300:
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
    if den < 1e-300:
        return np.nan
    return float(np.dot(aa, bb)/den)


def active_mask(y, th=0.02):
    p = np.max(np.abs(y))
    if p < 1e-300:
        return np.ones_like(y, dtype=bool)
    m = np.abs(y) > th*p
    if m.sum() < 10:
        m[:] = True
    return m


def compute_flux_channels(f, fS, vx, vy, vz, v2, w, rho, ux, T, qx, sig, R, Delta, Kn, chunk=32):
    nx = f.shape[0]

    Fq = np.zeros(nx)
    Qq = np.zeros(nx)
    Qd = np.zeros(nx)
    Delta_target = np.zeros(nx)

    for a in range(0, nx, chunk):
        b = min(nx, a+chunk)

        fb = f[a:b]
        fsb = fS[a:b]
        rb = rho[a:b]
        ub = ux[a:b]
        Tb = T[a:b]

        cx = vx[None, :] - ub[:, None]
        cy = vy[None, :]
        cz = vz[None, :]
        c2 = cx*cx + cy*cy + cz*cz
        c4 = c2*c2

        psi_q = 0.5*c2*cx

        Fq[a:b] = np.sum(w[None, :]*psi_q*vx[None, :]*fb, axis=1)
        Qq[a:b] = np.sum(w[None, :]*psi_q*(fsb-fb), axis=1)/Kn
        Qd[a:b] = np.sum(w[None, :]*c4*(fsb-fb), axis=1)/Kn

        # target scalar-excess moment relative to the same Maxwellian convention:
        # Delta_S = Delta + Kn Q_delta
        Delta_target[a:b] = Delta[a:b] + Kn*Qd[a:b]

    Bxx_from_flux = 2.0*(Fq - ux*qx)
    S_from_flux = Bxx_from_flux - (5.0*rho*T*T + 7.0*T*sig)
    S_from_flux = plateau_zero(S_from_flux)

    # True scalar-excess reconstruction from the Shakhov collision source.
    # If the Shakhov correction is c^4-neutral, then Q_delta = -Delta/Kn
    # and therefore Delta_source = -Kn*Q_delta.
    Delta_from_source = plateau_zero(-Kn*Qd)

    # Scalar excess carried by the Shakhov target itself:
    # Delta_target = Delta + Kn*Q_delta.  This should be close to zero
    # if the target has Maxwellian fourth scalar excess.
    Delta_target = plateau_zero(Delta_target)

    return Fq, Qq, Qd, Delta_target, S_from_flux, Delta_from_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullstate", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--outdir", default="figures/week2_jfm_upgrade_shakhov_cons")
    ap.add_argument("--Pr", type=float, default=2.0/3.0)
    ap.add_argument("--Kn", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-chunk", type=int, default=32)
    ap.add_argument("--moment-chunk", type=int, default=32)
    ap.add_argument("--active-thresh", type=float, default=0.02)
    args = ap.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device

    z = np.load(args.fullstate, allow_pickle=True)
    x_np = np.asarray(z[key(z, ["x", "x_mfp"])]).astype(np.float32)
    nx = len(x_np)

    vx_np, vy_np, vz_np, w_np = load_grid(z)
    nv = len(w_np)

    f_np = orient_f(z[key(z, ["f", "F", "dist", "distribution"])], nx, nv).astype(np.float32)

    vx = torch.tensor(vx_np, dtype=torch.float32, device=device)
    vy = torch.tensor(vy_np, dtype=torch.float32, device=device)
    vz = torch.tensor(vz_np, dtype=torch.float32, device=device)
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    v2 = vx*vx + vy*vy + vz*vz

    f = torch.tensor(f_np, dtype=torch.float32, device=device)

    fS, rho, ux, T, qx, sig, qS, m0err, m1err, eerr, min_target = build_target(
        f, vx, vy, vz, v2, w, args.Pr, args.target_chunk
    )

    R, Delta = high_moments_chunked(
        f, vx, vy, vz, v2, w,
        rho, ux, T, sig,
        chunk=args.moment_chunk
    )

    # Convert to numpy
    rho_n = rho.detach().cpu().numpy()
    ux_n = ux.detach().cpu().numpy()
    T_n = T.detach().cpu().numpy()
    qx_n = qx.detach().cpu().numpy()
    sig_n = sig.detach().cpu().numpy()
    R_n = plateau_zero(R.detach().cpu().numpy())
    Delta_n = plateau_zero(Delta.detach().cpu().numpy())
    S_n = plateau_zero(R_n + Delta_n/3.0)

    f_n = f.detach().cpu().numpy()
    fS_n = fS.detach().cpu().numpy()

    Fq, Qq, Qd, Delta_target, S_flux, Delta_src = compute_flux_channels(
        f_n, fS_n, vx_np, vy_np, vz_np, vx_np*vx_np+vy_np*vy_np+vz_np*vz_np,
        w_np, rho_n, ux_n, T_n, qx_n, sig_n,
        R_n, Delta_n, args.Kn, chunk=args.moment_chunk
    )

    R_delta0 = S_flux.copy()
    R_two = S_flux - Delta_src/3.0

    mask = active_mask(S_n, args.active_thresh)

    metrics = {
        "case": args.case,
        "Pr": args.Pr,
        "Kn": args.Kn,
        "S_flux_relL2_active": rel_l2(S_flux, S_n, mask),
        "S_flux_corr_active": corr(S_flux, S_n, mask),
        "Delta_source_relL2_active": rel_l2(Delta_src, Delta_n, mask),
        "Delta_source_corr_active": corr(Delta_src, Delta_n, mask),
        "R_delta0_relL2_active": rel_l2(R_delta0, R_n, mask),
        "R_two_channel_relL2_active": rel_l2(R_two, R_n, mask),
        "R_two_channel_corr_active": corr(R_two, R_n, mask),
        "mean_qS_over_q_active": float(np.nanmean(
            qS.detach().cpu().numpy()[mask] / (qx_n[mask] + 1e-300)
        )),
        "mean_Qq_over_q_active": float(np.nanmean(Qq[mask] / (qx_n[mask] + 1e-300))),
        "Delta_target_over_Delta_active": float(
            np.linalg.norm(Delta_target[mask]) / max(np.linalg.norm(Delta_n[mask]), 1e-300)
        ),
        "Delta_target_peak": float(np.max(np.abs(Delta_target))),
        "max_cons_mass": float(torch.max(torch.abs(m0err)).detach().cpu()),
        "max_cons_momx": float(torch.max(torch.abs(m1err)).detach().cpu()),
        "max_cons_energy": float(torch.max(torch.abs(eerr)).detach().cpu()),
        "min_f": float(torch.min(f).detach().cpu()),
        "min_fS": float(min_target.detach().cpu()),
    }

    outdir = Path(args.outdir)/args.case
    outdir.mkdir(parents=True, exist_ok=True)
    outstem = outdir/args.case

    pd.DataFrame([metrics]).to_csv(outstem.with_suffix(".metrics.csv"), index=False)

    pd.DataFrame({
        "x": x_np,
        "rho": rho_n,
        "ux": ux_n,
        "T": T_n,
        "qx": qx_n,
        "sigma_xx": sig_n,
        "R": R_n,
        "Delta": Delta_n,
        "S": S_n,
        "S_from_flux": S_flux,
        "Delta_from_source": Delta_src,
        "R_delta0": R_delta0,
        "R_two_channel": R_two,
        "Qq_direct": Qq,
        "Qdelta_direct": Qd,
        "Delta_target": Delta_target,
        "active_mask": mask.astype(int),
    }).to_csv(outstem.with_suffix(".profiles.csv"), index=False)

    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)

    axs[0].plot(x_np, S_n, "k-", label="direct S")
    axs[0].plot(x_np, S_flux, "r--", label="flux-channel S")
    axs[0].set_title(r"Kinematic heat-flux channel")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(fontsize=8)

    axs[1].plot(x_np, Delta_n, "k-", label=r"direct $\Delta$")
    axs[1].plot(x_np, Delta_src, "r--", label=r"source $\Delta$")
    axs[1].set_title(r"Shakhov scalar-excess source")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(fontsize=8)

    axs[2].plot(x_np, R_n, "k-", label=r"direct $R^{cl}_{xx}$")
    axs[2].plot(x_np, R_delta0, "r--", label=r"$S$, $\Delta=0$")
    axs[2].plot(x_np, R_two, "C2--", label="two-channel")
    axs[2].set_title(r"Two-channel Shakhov recovery")
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(fontsize=8)

    for ax in axs:
        ax.set_xlabel(r"$(x-x_s)/\lambda_1$")

    fig.savefig(str(outstem)+".pdf", bbox_inches="tight")
    fig.savefig(str(outstem)+".png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    print("\n=== metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:32s}: {v:.6g}")
        else:
            print(f"{k:32s}: {v}")

    print("\n[saved]")
    print(outstem.with_suffix(".metrics.csv"))
    print(outstem.with_suffix(".profiles.csv"))
    print(str(outstem)+".png")


if __name__ == "__main__":
    main()
