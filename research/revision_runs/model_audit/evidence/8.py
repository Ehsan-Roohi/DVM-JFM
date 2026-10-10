#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"None of these columns were found: {candidates}\n"
            f"Available columns are:\n{list(df.columns)}"
        )
    return None


def get_metric(dfm, name, default=None):
    if name in dfm.columns:
        return float(dfm.iloc[0][name])
    if default is not None:
        return default
    raise KeyError(f"Metric '{name}' not found in metrics CSV. Available: {list(dfm.columns)}")


def main():
    ap = argparse.ArgumentParser(
        description="Create final 3-panel Shakhov channel-consistency figure."
    )
    ap.add_argument("--profiles", required=True, help="Path to *.profiles.csv")
    ap.add_argument("--metrics", required=True, help="Path to *.metrics.csv")
    ap.add_argument("--out", required=True, help="Output figure path (.pdf or .png)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--panel-labels", action="store_true",
                    help="Add (a), (b), (c) labels")
    args = ap.parse_args()

    prof = pd.read_csv(args.profiles)
    met = pd.read_csv(args.metrics)

    # -----------------------------
    # robust column detection
    # -----------------------------
    xcol = pick_col(prof, [
        "x",
        "x_over_lambda1",
        "x_lambda1",
        "x_l1",
        "x_scaled",
    ])

    # panel (a): heat-flux channel
    Sref_col = pick_col(prof, [
        "S_ref",
        "S_reference",
        "S_dvm",
        "S",
    ])
    Sflux_col = pick_col(prof, [
        "S_flux",
        "S_from_flux",
        "S_rec",
        "S_reconstructed",
        "S_budget",
    ])

    # panel (b): scalar-excess channel
    Dref_col = pick_col(prof, [
        "Delta_ref",
        "Delta_reference",
        "Delta_dvm",
        "Delta",
    ])
    Dsrc_col = pick_col(prof, [
        "Delta_source",
        "Delta_from_source",
        "Delta_rec",
        "Delta_reconstructed",
        "Delta_budget",
        "Delta_target",
    ])

    # panel (c): R recovery
    Rref_col = pick_col(prof, [
        "R_ref",
        "R_reference",
        "R_dvm",
        "R",
        "Rcl_ref",
    ])
    R0_col = pick_col(prof, [
        "R_delta0",
        "R_from_S_only",
        "R_sonly",
        "R_Delta0",
        "R_noDelta",
    ])
    R2_col = pick_col(prof, [
        "R_two_channel",
        "R_two_channels",
        "R_from_two_channels",
        "R_twoBudget",
        "R_recovered",
    ])

    x = prof[xcol].to_numpy()
    Sref = prof[Sref_col].to_numpy()
    Sflux = prof[Sflux_col].to_numpy()
    Dref = prof[Dref_col].to_numpy()
    Dsrc = prof[Dsrc_col].to_numpy()
    Rref = prof[Rref_col].to_numpy()
    R0 = prof[R0_col].to_numpy()
    R2 = prof[R2_col].to_numpy()

    # -----------------------------
    # metrics
    # -----------------------------
    epsS = get_metric(met, "S_flux_relL2_active")
    epsD = get_metric(met, "Delta_source_relL2_active")
    epsR0 = get_metric(met, "R_delta0_relL2_active")
    epsR2 = get_metric(met, "R_two_channel_relL2_active")

    # correlation metric if available
    rhoR2 = None
    for cand in ["R_two_channel_corr_active", "R_twoBudget_corr_active", "R_rec_corr_active"]:
        if cand in met.columns:
            rhoR2 = float(met.iloc[0][cand])
            break

    # -----------------------------
    # plotting
    # -----------------------------
    fig, axs = plt.subplots(1, 3, figsize=(14.6, 4.8))
    plt.subplots_adjust(wspace=0.36, bottom=0.24, top=0.78)
    
    legend_kw = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        fontsize=8.5,
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=2.3,
        handletextpad=0.5
    )
    
    # -----------------------------
    # panel (a): S
    # -----------------------------
    ax = axs[0]
    ax.plot(x, Sref, color="k", lw=2.8, label=r"$S$ reference")
    ax.plot(x, Sflux, color="tab:red", lw=2.4, ls="--",
            label=r"$S$ from heat-flux channel")
    ax.set_xlabel(r"$x/\lambda_1$", fontsize=15)
    ax.set_ylabel(r"$S$", fontsize=16)
    ax.grid(alpha=0.25)
    ax.legend(**legend_kw)
    
    ax.text(
        0.05, 0.08, rf"$\epsilon_S = {epsS:.2e}$",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7")
    )
    
    if args.panel_labels:
        ax.text(0.50, -0.28, r"(a)", transform=ax.transAxes,
                ha="center", va="center", fontsize=16)
    
    # -----------------------------
    # panel (b): Delta
    # -----------------------------
    ax = axs[1]
    ax.plot(x, Dref, color="k", lw=2.8, label=r"$\Delta$ reference")
    ax.plot(x, Dsrc, color="tab:red", lw=2.4, ls="--",
            label=r"$\Delta$ from scalar channel")
    ax.set_xlabel(r"$x/\lambda_1$", fontsize=15)
    ax.set_ylabel(r"$\Delta$", fontsize=16)
    ax.grid(alpha=0.25)
    ax.legend(**legend_kw)
    
    ax.text(
        0.05, 0.08, rf"$\epsilon_\Delta = {epsD:.2e}$",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7")
    )
    
    if args.panel_labels:
        ax.text(0.50, -0.28, r"(b)", transform=ax.transAxes,
                ha="center", va="center", fontsize=16)
    
    # -----------------------------
    # panel (c): Rxx^cl
    # -----------------------------
    ax = axs[2]
    ax.plot(x, Rref, color="k", lw=2.9,
            label=r"$R^{\mathrm{cl}}_{xx}$ reference")
    ax.plot(x, R0, color="0.6", lw=2.4, ls=":",
            label=r"$R^{\mathrm{cl}}_{xx}$ from $S$ only")
    ax.plot(x, R2, color="tab:green", lw=2.6, ls="--",
            label=r"$R^{\mathrm{cl}}_{xx}$ from two channels")
    ax.set_xlabel(r"$x/\lambda_1$", fontsize=15)
    ax.set_ylabel(r"$R^{\mathrm{cl}}_{xx}$", fontsize=16)
    ax.grid(alpha=0.25)
    
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        fontsize=8.2,
        frameon=True,
        borderpad=0.25,
        labelspacing=0.20,
        handlelength=2.3,
        handletextpad=0.5
    )
    
    # text for metrics in panel (c)
    if rhoR2 is None:
        txt_c = (
            rf"$\epsilon_S = {epsS:.2e}$" "\n"
            rf"$\epsilon_\Delta = {epsD:.2e}$" "\n"
            rf"$\epsilon_R = {epsR2:.2e}$"
        )
    else:
        txt_c = (
            rf"$\epsilon_S = {epsS:.2e}$" "\n"
            rf"$\epsilon_\Delta = {epsD:.2e}$" "\n"
            rf"$\epsilon_R = {epsR2:.2e}$" "\n"
            rf"$\rho_R = {rhoR2:.6f}$"
        )
    
    # move metrics box OUTSIDE the third panel
    ax.text(
        1.03, 0.05, txt_c,
        transform=ax.transAxes,
        fontsize=11.5,
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7")
    )
    
    if args.panel_labels:
        ax.text(0.50, -0.28, r"(c)", transform=ax.transAxes,
                ha="center", va="center", fontsize=16)
    
    # -----------------------------
    # cosmetics
    # -----------------------------
    for ax in axs:
        ax.tick_params(labelsize=12)
    
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()