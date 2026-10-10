#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def local_poly_smooth_derivative(x, y, window=9, degree=3):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    n = len(x)
    window = int(window)
    degree = int(degree)

    if window % 2 == 0:
        window += 1
    window = max(window, degree + 2)
    if window % 2 == 0:
        window += 1

    half = window // 2

    ys = np.zeros(n)
    dy = np.zeros(n)

    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)

        # enlarge near boundaries if needed
        if b - a < degree + 1:
            if a == 0:
                b = min(n, degree + 1)
            elif b == n:
                a = max(0, n - degree - 1)

        z = x[a:b] - x[i]
        yy = y[a:b]

        V = np.vander(z, N=degree + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(V, yy, rcond=None)

        ys[i] = coef[0]
        dy[i] = coef[1] if degree >= 1 else 0.0

    return ys, dy


def rel_l2(a, b, mask):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    den = np.linalg.norm(b[mask])
    if den < 1e-300:
        return np.nan
    return float(np.linalg.norm(a[mask] - b[mask]) / den)


def corr(a, b, mask):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    aa = a[mask] - np.mean(a[mask])
    bb = b[mask] - np.mean(b[mask])
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den < 1e-300:
        return np.nan
    return float(np.dot(aa, bb) / den)


def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--active-thresh", type=float, default=0.02)
    ap.add_argument("--min-collisions", type=float, default=1.0)
    ap.add_argument("--out-prefix", default="dsmc_delta_budget_poly")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    require_cols(df, [
        "x", "n", "ux", "T", "Delta",
        "JDelta", "KDelta", "QDelta_collision",
        "collision_count"
    ])

    x = df["x"].to_numpy(float)
    n = df["n"].to_numpy(float)
    ux = df["ux"].to_numpy(float)
    T = df["T"].to_numpy(float)
    Delta = df["Delta"].to_numpy(float)
    J = df["JDelta"].to_numpy(float)
    K = df["KDelta"].to_numpy(float)
    Qcoll = df["QDelta_collision"].to_numpy(float)
    coll_count = df["collision_count"].to_numpy(float)

    J_s, dJdx = local_poly_smooth_derivative(x, J, args.window, args.degree)
    K_s, _ = local_poly_smooth_derivative(x, K, args.window, args.degree)
    ux_s, dudx = local_poly_smooth_derivative(x, ux, args.window, args.degree)

    Qbudget = dJdx + 4.0 * dudx * K_s
    residual = Qbudget - Qcoll

    peakD = np.max(np.abs(Delta))
    peakQ = np.max(np.abs(Qcoll))

    mask = np.ones_like(Delta, dtype=bool)
    if peakD > 1e-300:
        mask &= np.abs(Delta) > args.active_thresh * peakD
    if peakQ > 1e-300:
        mask |= np.abs(Qcoll) > args.active_thresh * peakQ

    mask &= coll_count >= args.min_collisions

    if mask.sum() < 10:
        mask = np.ones_like(Delta, dtype=bool)

    good = mask & (np.abs(Delta) > 1e-12 * max(1.0, peakD))
    if good.sum() > 5:
        nu_eff = -Qcoll[good] / Delta[good]
        nu_eff_mean = float(np.mean(nu_eff))
        nu_eff_median = float(np.median(nu_eff))
        nu_eff_std = float(np.std(nu_eff))
    else:
        nu_eff_mean = np.nan
        nu_eff_median = np.nan
        nu_eff_std = np.nan

    metrics = {
        "file": args.csv,
        "window": args.window,
        "degree": args.degree,
        "active_thresh": args.active_thresh,
        "n_active": int(mask.sum()),

        "Q_budget_vs_collision_relL2_active": rel_l2(Qbudget, Qcoll, mask),
        "Q_budget_vs_collision_corr_active": corr(Qbudget, Qcoll, mask),

        "Q_collision_vs_minusDelta_relL2_active": rel_l2(Qcoll, -Delta, mask),
        "Q_collision_vs_minusDelta_corr_active": corr(Qcoll, -Delta, mask),

        "residual_relL2_active": float(
            np.linalg.norm(residual[mask]) / max(np.linalg.norm(Qcoll[mask]), 1e-300)
        ),

        "Delta_peak": float(np.max(np.abs(Delta))),
        "Qcoll_peak": float(np.max(np.abs(Qcoll))),
        "Qbudget_peak": float(np.max(np.abs(Qbudget))),

        "mean_collision_count_active": float(np.mean(coll_count[mask])),
        "min_collision_count_active": float(np.min(coll_count[mask])),

        "nsamples": float(df["nsamples"].iloc[0]) if "nsamples" in df.columns else np.nan,
        "sample_time": float(df["sample_time"].iloc[0]) if "sample_time" in df.columns else np.nan,

        "nu_eff_mean_active": nu_eff_mean,
        "nu_eff_median_active": nu_eff_median,
        "nu_eff_std_active": nu_eff_std,
    }

    out = df.copy()
    out["JDelta_poly"] = J_s
    out["KDelta_poly"] = K_s
    out["ux_poly"] = ux_s
    out["dJdx_poly"] = dJdx
    out["dudx_poly"] = dudx
    out["QDelta_budget_poly"] = Qbudget
    out["QDelta_residual_poly"] = residual
    out["active_mask"] = mask.astype(int)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(str(out_prefix) + "_metrics.csv", index=False)
    out.to_csv(str(out_prefix) + "_profiles.csv", index=False)

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)

    ax[0, 0].plot(x, n, "k-")
    ax[0, 0].set_title("Density")
    ax[0, 0].grid(alpha=0.25)

    ax[0, 1].plot(x, T, "k-")
    ax[0, 1].set_title("Temperature")
    ax[0, 1].grid(alpha=0.25)

    ax[0, 2].plot(x, Delta, "k-")
    ax[0, 2].set_title(r"$\Delta$")
    ax[0, 2].grid(alpha=0.25)

    ax[1, 0].plot(x, Qcoll, "k-", label=r"collision $Q_\Delta$")
    ax[1, 0].plot(x, Qbudget, "r--", label=r"budget $Q_\Delta$")
    ax[1, 0].set_title(r"$Q_\Delta$: collision vs spatial budget")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.25)

    ax[1, 1].plot(x, residual, "C3-")
    ax[1, 1].set_title("Budget residual")
    ax[1, 1].grid(alpha=0.25)

    ax[1, 2].plot(x, Qcoll, "k-", label=r"$Q_\Delta^{DSMC}$")
    ax[1, 2].plot(x, -Delta, "C2--", label=r"$-\Delta$")
    ax[1, 2].set_title(r"Collision production vs $-\Delta$")
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(alpha=0.25)

    for a in ax.ravel():
        a.set_xlabel(r"$x/\lambda_1$")

    fig.savefig(str(out_prefix) + ".png", dpi=250, bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print("[metrics]")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("[done]", str(out_prefix) + "_metrics.csv")
    print("[done]", str(out_prefix) + "_profiles.csv")
    print("[done]", str(out_prefix) + ".png")
    print("[done]", str(out_prefix) + ".pdf")


if __name__ == "__main__":
    main()
