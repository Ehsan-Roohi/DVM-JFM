#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_col(df, names):
    for n in names:
        if n in df.columns:
            return df[n].to_numpy(dtype=float), n
    raise KeyError(f"Missing {names}; available={list(df.columns)}")


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


def active_mask(y, thresh=0.02):
    peak = np.max(np.abs(y))
    if peak < 1e-30:
        return np.ones_like(y, dtype=bool)
    m = np.abs(y) >= thresh*peak
    if m.sum() < 10:
        return np.ones_like(y, dtype=bool)
    return m


def select_sensors(x, y, n, mode="active", thresh=0.02, pad=0.25):
    if mode == "all":
        xs = np.linspace(x[0], x[-1], n)
    else:
        m = active_mask(y, thresh)
        xa = x[m]
        lo, hi = float(xa.min()), float(xa.max())
        dx = hi - lo
        lo = max(float(x[0]), lo - pad*dx)
        hi = min(float(x[-1]), hi + pad*dx)
        xs = np.linspace(lo, hi, n)

    xs = np.r_[x[0], xs, x[-1]]
    return np.unique(xs)


def rbf_matrix(x, centers, width):
    return np.exp(-0.5*((x[:, None]-centers[None, :])/width)**2)


def fit_rbf(x_eval, xs, ys, n_centers, width_mult, ridge, span=None, zero_plateau=True):
    if span is None:
        lo, hi = float(xs[1]), float(xs[-2])
    else:
        lo, hi = span

    centers = np.linspace(lo, hi, n_centers)
    spacing = max((hi-lo)/max(n_centers-1, 1), 1e-12)
    width = width_mult*spacing

    Phi = rbf_matrix(xs, centers, width)
    A = Phi.T@Phi + ridge*np.eye(n_centers)
    b = Phi.T@ys
    coef = np.linalg.solve(A, b)

    pred = rbf_matrix(x_eval, centers, width)@coef

    # Plateau-zeroing only makes sense for a full profile.
    # During leave-one-out cross validation x_eval is a single point,
    # so applying plateau_zero would either fail or erase the prediction.
    if zero_plateau and len(pred) >= 10:
        pred = plateau_zero(pred)

    return pred, centers, width, coef


def loo_score(xs, ys, n_centers, width_mult, ridge, span):
    errs = []
    for k in range(len(xs)):
        xtrain = np.delete(xs, k)
        ytrain = np.delete(ys, k)
        pred, _, _, _ = fit_rbf(
            xs[[k]], xtrain, ytrain,
            n_centers=n_centers,
            width_mult=width_mult,
            ridge=ridge,
            span=span,
            zero_plateau=False,
        )
        scale = max(np.max(np.abs(ys)), 1e-12)
        errs.append(((pred[0]-ys[k])/scale)**2)
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--outdir", default="figures/week1_jfm_upgrade_delta_rbf")

    ap.add_argument("--n-delta", type=int, default=24)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--sensor-mode", choices=["active", "all"], default="active")
    ap.add_argument("--active-thresh", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--ncenters", default="auto",
                    help="'auto' or integer number of RBF centres")
    ap.add_argument("--width-mult", default="auto",
                    help="'auto' or width multiplier")
    ap.add_argument("--ridge", default="auto",
                    help="'auto' or ridge value")

    args = ap.parse_args()

    outdir = Path(args.outdir)/args.case
    outdir.mkdir(parents=True, exist_ok=True)
    outstem = outdir/args.case

    df = pd.read_csv(args.profiles)
    x, _ = get_col(df, ["x"])
    R_ref, rname = get_col(df, ["Rxx_closure", "R", "Rxx", "R_ref"])
    Delta_ref, dname = get_col(df, ["Delta", "Delta_true", "Delta_ref"])

    if "S_R_plus_Delta3_rec" in df.columns:
        S_budget = df["S_R_plus_Delta3_rec"].to_numpy(dtype=float)
        sname = "S_R_plus_Delta3_rec"
    else:
        S_budget, sname = get_col(df, ["S_R_plus_Delta3_ref", "S", "S_true", "S_target"])

    R_ref = plateau_zero(R_ref)
    Delta_ref = plateau_zero(Delta_ref)
    S_budget = plateau_zero(S_budget)

    R_no_delta = S_budget.copy()
    R_true_delta = S_budget - Delta_ref/3.0

    rng = np.random.default_rng(args.seed)
    xs = select_sensors(x, Delta_ref, args.n_delta, args.sensor_mode, args.active_thresh)
    dclean = np.interp(xs, x, Delta_ref)
    dnoisy = dclean.copy()

    interior = (xs > x[0]+1e-12) & (xs < x[-1]-1e-12)
    amp = max(np.max(np.abs(dclean[interior])) if interior.any() else np.max(np.abs(dclean)), 1e-12)
    dnoisy[interior] += args.noise*amp*rng.standard_normal(interior.sum())

    span = (float(xs[1]), float(xs[-2]))

    if args.ncenters == "auto":
        ncent_list = sorted(set([
            max(args.n_delta+4, 8),
            max(args.n_delta+8, 10),
            max(2*args.n_delta, 12),
        ]))
    else:
        ncent_list = [int(args.ncenters)]

    if args.width_mult == "auto":
        wm_list = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
    else:
        wm_list = [float(args.width_mult)]

    if args.ridge == "auto":
        ridge_list = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    else:
        ridge_list = [float(args.ridge)]

    # Select hyperparameters using only sparse Delta data through leave-one-out.
    best = None
    for nc in ncent_list:
        for wm in wm_list:
            for rg in ridge_list:
                try:
                    score = loo_score(xs, dnoisy, nc, wm, rg, span)
                except np.linalg.LinAlgError:
                    continue
                item = (score, nc, wm, rg)
                if best is None or item < best:
                    best = item

    if best is None:
        raise RuntimeError("RBF hyperparameter search failed.")

    loo, ncent, wmult, ridge = best
    Delta_rbf, centers, width, coef = fit_rbf(
        x, xs, dnoisy,
        n_centers=ncent,
        width_mult=wmult,
        ridge=ridge,
        span=span,
    )

    Delta_interp = plateau_zero(np.interp(x, xs, dnoisy))

    R_interp = S_budget - Delta_interp/3.0
    R_rbf = S_budget - Delta_rbf/3.0

    mask = active_mask(S_budget, args.active_thresh)

    metrics = {
        "case": args.case,
        "profiles": args.profiles,
        "R_col": rname,
        "Delta_col": dname,
        "S_col": sname,
        "n_delta": args.n_delta,
        "noise": args.noise,
        "active_thresh": args.active_thresh,
        "ncenters": ncent,
        "width_mult": wmult,
        "width": width,
        "ridge": ridge,
        "loo_score": loo,

        "R_no_delta_relL2_active": rel_l2(R_no_delta, R_ref, mask),
        "R_true_delta_relL2_active": rel_l2(R_true_delta, R_ref, mask),
        "Delta_interp_relL2_active": rel_l2(Delta_interp, Delta_ref, mask),
        "Delta_rbf_relL2_active": rel_l2(Delta_rbf, Delta_ref, mask),
        "R_interp_relL2_active": rel_l2(R_interp, R_ref, mask),
        "R_rbf_relL2_active": rel_l2(R_rbf, R_ref, mask),
        "R_rbf_corr_active": corr(R_rbf, R_ref, mask),
        "Delta_rbf_corr_active": corr(Delta_rbf, Delta_ref, mask),

        "R_no_delta_relL2_full": rel_l2(R_no_delta, R_ref),
        "R_true_delta_relL2_full": rel_l2(R_true_delta, R_ref),
        "Delta_interp_relL2_full": rel_l2(Delta_interp, Delta_ref),
        "Delta_rbf_relL2_full": rel_l2(Delta_rbf, Delta_ref),
        "R_interp_relL2_full": rel_l2(R_interp, R_ref),
        "R_rbf_relL2_full": rel_l2(R_rbf, R_ref),
    }

    pd.DataFrame([metrics]).to_csv(outstem.with_suffix(".metrics.csv"), index=False)

    pd.DataFrame({
        "x": x,
        "R_ref": R_ref,
        "Delta_ref": Delta_ref,
        "S_budget": S_budget,
        "R_no_delta": R_no_delta,
        "Delta_interp": Delta_interp,
        "Delta_rbf": Delta_rbf,
        "R_interp": R_interp,
        "R_rbf": R_rbf,
        "R_true_delta": R_true_delta,
        "active_mask": mask.astype(int),
    }).to_csv(outstem.with_suffix(".profiles.csv"), index=False)

    pd.DataFrame({
        "x_sensor": xs,
        "Delta_clean": dclean,
        "Delta_noisy": dnoisy,
    }).to_csv(outstem.with_suffix(".sensors.csv"), index=False)

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)

    axs[0].plot(x, Delta_ref, "k-", lw=1.8, label=r"DVM $\Delta$")
    axs[0].plot(x, Delta_interp, "--", color="0.55", lw=1.4, label="linear interp.")
    axs[0].plot(x, Delta_rbf, "C2-", lw=1.6, label="RBF channel")
    axs[0].plot(xs, dnoisy, "o", ms=3.5, label="sparse probes")
    axs[0].set_title(r"Second channel: sparse $\Delta$")
    axs[0].set_xlabel(r"$(x-x_s)/\lambda_1$")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(fontsize=8, frameon=True)

    axs[1].plot(x, R_ref, "k-", lw=1.9, label=r"DVM $R^{cl}_{xx}$")
    axs[1].plot(x, R_no_delta, "r--", lw=1.6, label=r"$S$ only, $\Delta=0$")
    axs[1].set_title("Rank-deficient heat-flux channel")
    axs[1].set_xlabel(r"$(x-x_s)/\lambda_1$")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(fontsize=8, frameon=True)

    axs[2].plot(x, R_ref, "k-", lw=1.9, label=r"DVM $R^{cl}_{xx}$")
    axs[2].plot(x, R_interp, "--", color="0.55", lw=1.4, label=r"$S-\Delta_{\rm interp}/3$")
    axs[2].plot(x, R_rbf, "C2-", lw=1.6, label=r"$S-\Delta_{\rm RBF}/3$")
    axs[2].set_title("Recovered tensorial closure")
    axs[2].set_xlabel(r"$(x-x_s)/\lambda_1$")
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(fontsize=8, frameon=True)

    fig.suptitle(args.case + r": resolving $R^{cl}_{xx}$ with a sparse scalar-excess channel", fontsize=12)
    fig.text(
        0.5, -0.02,
        f"R error: Δ=0 {100*metrics['R_no_delta_relL2_active']:.1f}%   "
        f"true Δ {100*metrics['R_true_delta_relL2_active']:.2f}%   "
        f"sparse Δ-RBF {100*metrics['R_rbf_relL2_active']:.2f}%",
        ha="center", va="top", fontsize=10
    )

    fig.savefig(str(outstem)+".pdf", bbox_inches="tight")
    fig.savefig(str(outstem)+".png", dpi=350, bbox_inches="tight")
    plt.close(fig)

    print("\n=== metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:35s}: {v:.6g}")
        else:
            print(f"{k:35s}: {v}")
    print("\n[saved]")
    print(outstem.with_suffix(".metrics.csv"))
    print(outstem.with_suffix(".profiles.csv"))
    print(outstem.with_suffix(".sensors.csv"))
    print(str(outstem)+".pdf")
    print(str(outstem)+".png")


if __name__ == "__main__":
    main()
