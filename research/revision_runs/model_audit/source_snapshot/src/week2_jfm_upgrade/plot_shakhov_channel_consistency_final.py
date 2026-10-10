#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns found: {candidates}\nAvailable: {list(df.columns)}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory containing <case>.profiles.csv and <case>.metrics.csv")
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True, help="Output figure path (.pdf recommended)")
    args = ap.parse_args()

    indir = Path(args.indir)
    case = args.case

    p_prof = indir / f"{case}.profiles.csv"
    p_met  = indir / f"{case}.metrics.csv"

    df = pd.read_csv(p_prof)
    met = pd.read_csv(p_met).iloc[0] if p_met.exists() else None

    # ---- column detection ----
    xcol = pick(df, ["x_mfp", "x", "x_center", "x_raw"])

    Sref_col  = pick(df, ["S_ref", "S", "S_true", "S_target", "S_direct", "S_exact"])
    Sflux_col = pick(df, ["S_flux", "S_rec", "S_budget", "S_from_flux", "S_recovered"])

    Dref_col  = pick(df, ["Delta_target", "Delta_ref", "Delta", "Delta_true", "Delta_exact"])
    Dsrc_col  = pick(df, ["Delta_source", "Delta_rec", "Delta_budget", "Delta_from_source", "Delta_recovered"])

    Rref_col  = pick(df, ["Rxx_closure", "R_ref", "R_true", "Rxx_closure_ref", "R"])
    R0_col    = pick(df, ["R_delta0", "R_rec_Delta0", "R_from_S_only", "R_without_Delta"])
    R2_col    = pick(df, ["R_two_channel", "R_twoBudget", "R_with_Delta", "R_rec_with_Delta"])

    x = df[xcol].to_numpy()

    Sref  = df[Sref_col].to_numpy()
    Sflux = df[Sflux_col].to_numpy()

    Dref  = df[Dref_col].to_numpy()
    Dsrc  = df[Dsrc_col].to_numpy()

    Rref = df[Rref_col].to_numpy()
    R0   = df[R0_col].to_numpy()
    R2   = df[R2_col].to_numpy()

    # ---- styling ----
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
    })

    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)

    # ---------------- panel (a): S ----------------
    ax = axs[0]
    ax.plot(x, Sref,  color="k",        lw=2.2, label=r"$S$ reference")
    ax.plot(x, Sflux, color="tab:red",  lw=1.9, ls="--", label=r"$S_{\mathrm{flux}}$")
    ax.set_xlabel(r"$x/\lambda_1$")
    ax.set_ylabel(r"$S$")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, handlelength=2.4, borderpad=0.35, labelspacing=0.25)

    # ---------------- panel (b): Delta ----------------
    ax = axs[1]
    ax.plot(x, Dref, color="k",        lw=2.2, label=r"$\Delta$ target")
    ax.plot(x, Dsrc, color="tab:blue", lw=1.9, ls="--", label=r"$\Delta_{\mathrm{source}}$")
    ax.set_xlabel(r"$x/\lambda_1$")
    ax.set_ylabel(r"$\Delta$")
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(loc="best", frameon=True, handlelength=2.4, borderpad=0.35, labelspacing=0.25)

    # ---------------- panel (c): R ----------------
    ax = axs[2]
    ax.plot(x, Rref, color="k",         lw=2.2, label=r"$R^{\mathrm{cl}}_{xx}$ reference")
    ax.plot(x, R0,   color="0.55",      lw=1.9, ls=":",  label=r"$R^{\mathrm{cl}}_{xx}$ from $S$ only ($\Delta=0$)")
    ax.plot(x, R2,   color="tab:green", lw=1.9, ls="--", label=r"$R^{\mathrm{cl}}_{xx}$ from two channels")
    ax.set_xlabel(r"$x/\lambda_1$")
    ax.set_ylabel(r"$R^{\mathrm{cl}}_{xx}$")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True, handlelength=2.6, borderpad=0.35, labelspacing=0.25)

    # panel labels below each frame
    for i, lab in enumerate(["(a)", "(b)", "(c)"]):
        axs[i].text(0.5, -0.30, lab, transform=axs[i].transAxes,
                    ha="center", va="top", fontsize=11)

    # optional compact metrics text
    if met is not None:
        txt = []
        if "S_flux_relL2_active" in met:
            txt.append(r"$\varepsilon_S={:.2e}$".format(float(met["S_flux_relL2_active"])))
        if "Delta_source_relL2_active" in met:
            txt.append(r"$\varepsilon_\Delta={:.2e}$".format(float(met["Delta_source_relL2_active"])))
        if "R_two_channel_relL2_active" in met:
            txt.append(r"$\varepsilon_R={:.2e}$".format(float(met["R_two_channel_relL2_active"])))
        if txt:
            axs[2].text(0.03, 0.04, "\n".join(txt),
                        transform=axs[2].transAxes,
                        ha="left", va="bottom", fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if out.suffix.lower() != ".png":
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[done]", out)
    if out.suffix.lower() != ".png":
        print("[done]", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
