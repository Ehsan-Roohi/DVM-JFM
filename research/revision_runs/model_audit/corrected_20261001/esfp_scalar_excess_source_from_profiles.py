#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_col(df, candidates, required=True):
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(
            f"Could not find any of {candidates}\nAvailable columns:\n{list(df.columns)}"
        )
    return None


def rel_l2(a, b, mask):
    den = np.linalg.norm(b[mask])
    if den < 1e-300:
        return np.nan
    return float(np.linalg.norm(a[mask] - b[mask]) / den)


def corr(a, b, mask):
    aa = a[mask] - np.mean(a[mask])
    bb = b[mask] - np.mean(b[mask])
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den < 1e-300:
        return np.nan
    return float(np.dot(aa, bb) / den)


def active_mask(y, thresh=0.02):
    peak = np.max(np.abs(y))
    if peak < 1e-300:
        return np.ones_like(y, dtype=bool)
    m = np.abs(y) > thresh * peak
    if m.sum() < 10:
        m[:] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True, help="DVM/BGK shock profiles CSV")
    ap.add_argument("--case", required=True)
    ap.add_argument("--outdir", default="figures/esfp_scalar_source")
    ap.add_argument("--Kn", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=-1.25,
                    help="ES-FP beta. For correct Prandtl number often beta=-5/4.")
    ap.add_argument("--nu-factor", type=float, default=1.0,
                    help="Dimensionless collision-frequency multiplier. Source uses nu=nu_factor/Kn.")
    ap.add_argument("--active-thresh", type=float, default=0.02)
    ap.add_argument("--esbgk-alpha", type=float, default=-0.5,
                    help="Optional ES-BGK alpha=1-1/Pr for comparison.")
    args = ap.parse_args()

    df = pd.read_csv(args.profiles)

    xcol = pick_col(df, ["x", "x_mfp", "x_center", "x_raw"])
    ncol = pick_col(df, ["rho_from_f", "rho_file", "n", "rho", "density"])
    Tcol = pick_col(df, ["T_from_f", "T_file", "T", "temperature", "Temp"])
    sigcol = pick_col(df, ["sigma_xx", "sigxx", "sig_xx", "stress_xx", "sigma"])
    Dcol = pick_col(df, ["Delta", "delta", "Delta_DVM", "DVM_Delta"])

    x = df[xcol].to_numpy(dtype=float)
    n = df[ncol].to_numpy(dtype=float)
    T = df[Tcol].to_numpy(dtype=float)
    sig = df[sigcol].to_numpy(dtype=float)
    Delta = df[Dcol].to_numpy(dtype=float)

    # 1D normal shock: traceless stress tensor
    # sigma = diag(sigma_xx, -sigma_xx/2, -sigma_xx/2)
    tr_sig2 = 1.5 * sig * sig

    nu = args.nu_factor / args.Kn

    # Unscaled source per physical/collision frequency:
    # Q_Delta^FP = nu[-4 Delta + 8 beta tr(sigma^2)/n]
    fp_base_unit = -4.0 * nu * Delta
    esfp_corr_unit = 8.0 * nu * args.beta * tr_sig2 / np.maximum(n, 1e-300)
    esfp_source_unit = fp_base_unit + esfp_corr_unit

    # Scaled source, comparable to BGK/Shakhov -Delta/Kn shape after division by 4nu.
    # Q/(4nu) = -Delta + 2 beta tr(sigma^2)/n.
    bgk_like = -Delta
    esfp_corr_scaled = 2.0 * args.beta * tr_sig2 / np.maximum(n, 1e-300)
    esfp_source_scaled = bgk_like + esfp_corr_scaled

    # ES-BGK comparison:
    # Delta_G^ES = 2 alpha^2 tr(sigma^2)/n
    esbgk_deltaG = 2.0 * args.esbgk_alpha**2 * tr_sig2 / np.maximum(n, 1e-300)
    esbgk_source_scaled = bgk_like + esbgk_deltaG

    mask = active_mask(Delta, args.active_thresh)

    metrics = {
        "case": args.case,
        "Kn": args.Kn,
        "nu_factor": args.nu_factor,
        "beta_ESFP": args.beta,
        "alpha_ESBGK": args.esbgk_alpha,
        "active_thresh": args.active_thresh,
        "n_active": int(mask.sum()),

        "Delta_peak": float(np.max(np.abs(Delta))),
        "sigma_xx_peak": float(np.max(np.abs(sig))),
        "tr_sigma2_peak": float(np.max(np.abs(tr_sig2))),

        "ESFP_corr_scaled_peak": float(np.max(np.abs(esfp_corr_scaled))),
        "ESFP_corr_over_Delta_active": float(
            np.linalg.norm(esfp_corr_scaled[mask]) / max(np.linalg.norm(Delta[mask]), 1e-300)
        ),
        "ESFP_corr_over_Delta_active_direct": float(
            np.linalg.norm(esfp_corr_scaled[mask]) / max(np.linalg.norm(Delta[mask]), 1e-300)
        ),

        "ESFP_scaled_vs_BGK_relL2_active": rel_l2(esfp_source_scaled, bgk_like, mask),
        "ESFP_scaled_vs_BGK_corr_active": corr(esfp_source_scaled, bgk_like, mask),

        "ESFP_unit_vs_FP_relL2_active": rel_l2(esfp_source_unit, fp_base_unit, mask),
        "ESFP_unit_vs_FP_corr_active": corr(esfp_source_unit, fp_base_unit, mask),

        "ESBGK_corr_over_Delta_active": float(
            np.linalg.norm(esbgk_deltaG[mask]) / max(np.linalg.norm(Delta[mask]), 1e-300)
        ),

        "ESFPcorr_over_ESBGKcorr_active_signed": float(
            np.dot(esfp_corr_scaled[mask], esbgk_deltaG[mask]) /
            max(np.dot(esbgk_deltaG[mask], esbgk_deltaG[mask]), 1e-300)
        ),
    }

    outdir = Path(args.outdir) / args.case
    outdir.mkdir(parents=True, exist_ok=True)
    outstem = outdir / args.case

    out = pd.DataFrame({
        "x": x,
        "n": n,
        "T": T,
        "sigma_xx": sig,
        "tr_sigma2": tr_sig2,
        "Delta": Delta,
        "Q_FP_unit": fp_base_unit,
        "Q_ESFP_unit": esfp_source_unit,
        "Q_ESFP_corr_unit": esfp_corr_unit,
        "Q_BGK_like_scaled": bgk_like,
        "Q_ESFP_scaled": esfp_source_scaled,
        "Q_ESFP_corr_scaled": esfp_corr_scaled,
        "DeltaG_ESBGK": esbgk_deltaG,
        "Q_ESBGK_scaled": esbgk_source_scaled,
        "active_mask": mask.astype(int),
    })

    out.to_csv(outstem.with_suffix(".csv"), index=False)
    pd.DataFrame([metrics]).to_csv(outstem.with_name(outstem.name + "_metrics.csv"), index=False)

    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)

    axs[0].plot(x, Delta, "k-", label=r"$\Delta$")
    axs[0].plot(x, esfp_corr_scaled, "r--", label=r"ES-FP correction")
    axs[0].plot(x, esbgk_deltaG, "C2:", label=r"ES-BGK correction")
    axs[0].set_title(r"Scalar-excess source corrections")
    axs[0].set_xlabel(r"$x/\lambda_1$")
    axs[0].grid(alpha=0.25)
    axs[0].legend(fontsize=8)

    axs[1].plot(x, bgk_like, "k-", label=r"BGK/Shakhov: $-\Delta$")
    axs[1].plot(x, esfp_source_scaled, "r--", label=r"ES-FP scaled source")
    axs[1].plot(x, esbgk_source_scaled, "C2:", label=r"ES-BGK scaled source")
    axs[1].set_title(r"Scaled \(Q_\Delta\) source")
    axs[1].set_xlabel(r"$x/\lambda_1$")
    axs[1].grid(alpha=0.25)
    axs[1].legend(fontsize=8)

    local_ratio = np.zeros_like(Delta)
    good = np.abs(Delta) > args.active_thresh * max(np.max(np.abs(Delta)), 1e-300)
    local_ratio[good] = esfp_corr_scaled[good] / Delta[good]

    axs[2].plot(x, local_ratio, "r-", label=r"ES-FP correction$/\Delta$")
    axs[2].axhline(0.0, color="0.5", lw=1.0)
    axs[2].set_title(r"Local correction ratio")
    axs[2].set_xlabel(r"$x/\lambda_1$")
    axs[2].grid(alpha=0.25)
    axs[2].legend(fontsize=8)

    fig.savefig(str(outstem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(outstem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("=== ES-FP scalar-excess source diagnostic ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("[done]", outstem.with_suffix(".csv"))
    print("[done]", outstem.with_name(outstem.name + "_metrics.csv"))
    print("[done]", str(outstem) + ".pdf")
    print("[done]", str(outstem) + ".png")


if __name__ == "__main__":
    main()
