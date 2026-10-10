#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def key(z, names):
    for n in names:
        if n in z.files:
            return n
    raise KeyError(f"Missing {names}; available={z.files}")


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
    raise ValueError(f"Cannot orient f shape={f.shape}, nx={nx}, nv={nv}")


def plateau_zero(y, frac=0.08):
    y = np.asarray(y, float).copy()
    n = len(y)
    ne = max(5, int(frac*n))
    idx = np.r_[np.arange(ne), np.arange(n-ne, n)]
    return y - np.mean(y[idx])


def moving_average(y, window):
    y = np.asarray(y, float)
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window//2
    yp = np.pad(y, (pad, pad), mode="edge")
    k = np.ones(window)/window
    return np.convolve(yp, k, mode="valid")


def rel(a, b, mask):
    den = np.linalg.norm(b[mask])
    return np.linalg.norm(a[mask]-b[mask]) / max(den, 1e-300)


def corr(a, b, mask):
    aa = a[mask] - np.mean(a[mask])
    bb = b[mask] - np.mean(b[mask])
    return np.dot(aa, bb)/max(np.linalg.norm(aa)*np.linalg.norm(bb), 1e-300)


def active(y, th=0.02):
    p = np.max(np.abs(y))
    m = np.abs(y) > th*p
    if m.sum() < 10:
        m[:] = True
    return m


def load_grid(z):
    v = np.asarray(z[key(z, ["v", "vel", "velocity", "velocities"])])
    if v.ndim == 2 and v.shape[1] >= 3:
        vx, vy, vz = v[:,0], v[:,1], v[:,2]
    elif v.ndim == 2 and v.shape[0] >= 3:
        vx, vy, vz = v[0], v[1], v[2]
    else:
        raise ValueError(f"Cannot parse v shape {v.shape}")
    w = np.asarray(z[key(z, ["w", "weights", "weight", "vw"])]).ravel()
    return vx.ravel().astype(float), vy.ravel().astype(float), vz.ravel().astype(float), w.astype(float)


def compute_chunk(f, vx, vy, vz, w, rho, ux, T, Pr, Kn, coeff_factor):
    cx = vx[None,:] - ux[:,None]
    cy = vy[None,:]
    cz = vz[None,:]
    c2 = cx*cx + cy*cy + cz*cz

    M = rho[:,None]/((2*np.pi*T[:,None])**1.5)*np.exp(-0.5*c2/T[:,None])

    psi = 0.5*c2*cx
    q = np.sum(w[None,:]*psi*f, axis=1)

    # Shakhov target with adjustable coefficient factor.
    # coeff_factor=1 is standard if q = 1/2 int c^2 c_x f dv.
    phi = coeff_factor*(1-Pr)*(cx*q[:,None])/(5*rho[:,None]*T[:,None]**2 + 1e-300)*(c2/T[:,None]-5)
    fS = M*(1+phi)

    qS = np.sum(w[None,:]*psi*fS, axis=1)
    Qdirect = np.sum(w[None,:]*psi*(fS-f), axis=1)/Kn

    sig = np.sum(w[None,:]*(cx*cx-c2/3)*f, axis=1)
    kr = c2*(cx*cx-c2/3)
    R = np.sum(w[None,:]*kr*(f-M), axis=1) - 7*T*sig
    Delta = np.sum(w[None,:]*c2*c2*(f-M), axis=1)
    S = R + Delta/3

    L = ux*q + 0.5*(5*rho*T*T + 7*T*sig)

    # Heat-flux geometric helper:
    # Hq = int v_x (c_x^2 + 0.5 |c|^2) f dv.
    Hq = np.sum(w[None,:]*vx[None,:]*(cx*cx + 0.5*c2)*f, axis=1)

    return q, qS, Qdirect, R, Delta, S, L, Hq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullstate", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--outdir", default="figures/week2_jfm_upgrade_shakhov")
    ap.add_argument("--Pr", type=float, default=2/3)
    ap.add_argument("--Kn", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--smooth-deriv", type=int, default=7)
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

    results = {}

    for cf in [0.5, 1.0, 2.0]:
        q = np.zeros(len(x))
        qS = np.zeros(len(x))
        Qd = np.zeros(len(x))
        R = np.zeros(len(x))
        D = np.zeros(len(x))
        S = np.zeros(len(x))
        L = np.zeros(len(x))
        Hq = np.zeros(len(x))

        for a in range(0, len(x), args.chunk):
            b = min(len(x), a+args.chunk)
            out = compute_chunk(
                f[a:b], vx, vy, vz, w,
                rho[a:b], ux[a:b], T[a:b],
                args.Pr, args.Kn, cf
            )
            q[a:b], qS[a:b], Qd[a:b], R[a:b], D[a:b], S[a:b], L[a:b], Hq[a:b] = out

        R = plateau_zero(R)
        D = plateau_zero(D)
        S = plateau_zero(S)

        # Reconstruct S from direct collision source.
        # Include the moving-frame geometric term from d/dx psi_q.
        dLdx = np.gradient(moving_average(L, args.smooth_deriv), x)
        dudx = np.gradient(moving_average(ux, args.smooth_deriv), x)
        heat_geom = dudx * moving_average(Hq, args.smooth_deriv)
        dSdx = 2*(Qd - heat_geom - dLdx)

        Srec = np.zeros_like(S)
        Srec[1:] = np.cumsum(0.5*(dSdx[1:]+dSdx[:-1])*np.diff(x))
        Srec = plateau_zero(Srec)

        m = active(S, 0.02)

        results[cf] = dict(
            q=q, qS=qS, Qd=Qd, R=R, D=D, S=S, L=L, Srec=Srec,
            S_rel=rel(Srec, S, m),
            S_corr=corr(Srec, S, m),
            mean_qS_over_q=np.nanmean(qS[m]/(q[m]+1e-300)),
            mean_Q_over_q=np.nanmean(Qd[m]/(q[m]+1e-300)),
        )

    rows = []
    for cf, r in results.items():
        rows.append({
            "coeff_factor": cf,
            "S_rec_relL2_active": r["S_rel"],
            "S_rec_corr_active": r["S_corr"],
            "mean_qS_over_q_active": r["mean_qS_over_q"],
            "mean_Qdirect_over_q_active": r["mean_Q_over_q"],
            "expected_qS_over_q": cf*(1-args.Pr),
            "expected_Q_over_q": cf*(1-args.Pr)-1,
        })

    pd.DataFrame(rows).to_csv(outstem.with_suffix(".metrics.csv"), index=False)

    # Plot.
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)

    axs[0].plot(x, results[1.0]["S"], "k-", label="DVM S")
    for cf, style in [(0.5, "C0--"), (1.0, "r--"), (2.0, "C2--")]:
        axs[0].plot(x, results[cf]["Srec"], style, label=f"Srec cf={cf}")
    axs[0].set_title("S reconstruction from direct Shakhov source")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(fontsize=8)

    axs[1].plot(x, results[1.0]["q"], "k-", label="q")
    for cf, style in [(0.5, "C0--"), (1.0, "r--"), (2.0, "C2--")]:
        axs[1].plot(x, results[cf]["qS"], style, label=f"q target cf={cf}")
    axs[1].set_title("Heat flux of Shakhov target")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(fontsize=8)

    axs[2].plot(x, results[1.0]["Qd"], "r-", label="Q direct cf=1")
    axs[2].plot(x, -args.Pr*results[1.0]["q"]/args.Kn, "k--", label="-Pr q/Kn")
    axs[2].plot(x, -results[1.0]["q"]/args.Kn, linestyle=":", color="0.5", label="-q/Kn")
    axs[2].set_title("Collision source")
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(fontsize=8)

    for ax in axs:
        ax.set_xlabel(r"$(x-x_s)/\lambda_1$")

    fig.savefig(str(outstem)+".pdf", bbox_inches="tight")
    fig.savefig(str(outstem)+".png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    print(pd.DataFrame(rows).to_string(index=False))
    print("[done]", outstem.with_suffix(".metrics.csv"))
    print("[done]", str(outstem)+".png")


if __name__ == "__main__":
    main()
