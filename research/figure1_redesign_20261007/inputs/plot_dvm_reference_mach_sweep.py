#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def read_profile_csv(path):
    df = pd.read_csv(path)

    required = ["x", "qx", "Rxx_closure", "Delta", "S_R_plus_Delta3_ref"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns {miss} in {path}\nAvailable: {list(df.columns)}")

    x = df["x"].to_numpy(dtype=float)

    # اگر rho موجود بود، shock center را از بیشینه گرادیان چگالی می‌گیریم
    # اگر نبود، فرض می‌کنیم x از قبل centered است.
    if "rho_file" in df.columns:
        rho = df["rho_file"].to_numpy(dtype=float)
        drho = np.gradient(rho, x)
        xs = x[np.argmax(np.abs(drho))]
    else:
        xs = 0.0

    xhat = x - xs

    out = {
        "x": xhat,
        "qx": df["qx"].to_numpy(dtype=float),
        "R": df["Rxx_closure"].to_numpy(dtype=float),
        "Delta": df["Delta"].to_numpy(dtype=float),
        "S": df["S_R_plus_Delta3_ref"].to_numpy(dtype=float),
    }
    return out


def add_panel_label(ax, txt):
    ax.text(
        0.04, 0.06, txt,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=11, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.2)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2", required=True, help="M2 exact-budget xmfp csv")
    ap.add_argument("--m3", required=True, help="M3 exact-budget xmfp csv")
    ap.add_argument("--m5", required=True, help="M5 exact-budget xmfp csv")
    ap.add_argument("--out", required=True, help="output stem without extension")
    args = ap.parse_args()

    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "mathtext.fontset": "dejavuserif",
        "font.family": "serif",
        "axes.linewidth": 1.0,
    })

    cases = {
        "M2": read_profile_csv(args.m2),
        "M3": read_profile_csv(args.m3),
        "M5": read_profile_csv(args.m5),
    }

    row_keys = ["qx", "R", "Delta", "S"]
    row_labels = [
        r"$q_x$",
        r"$R_{xx}^{cl}$",
        r"$\Delta$",
        r"$S = R + \Delta/3$",
    ]
    col_titles = ["M2", "M3", "M5"]

    fig, axs = plt.subplots(
        4, 3, figsize=(10.2, 10.8),
        sharex=False
    )

    panel_labels = list("abcdefghijkl")

    for j, M in enumerate(col_titles):
        d = cases[M]
        x = d["x"]

        series = [d["qx"], d["R"], d["Delta"], d["S"]]

        for i in range(4):
            ax = axs[i, j]
            y = series[i]

            ax.plot(x, y, color="k", lw=2.0)

            # کمی margin عمودی
            y_min = np.nanmin(y)
            y_max = np.nanmax(y)
            if np.isfinite(y_min) and np.isfinite(y_max):
                if abs(y_max - y_min) < 1e-14:
                    pad = 1.0
                else:
                    pad = 0.12 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)

            ax.grid(True, alpha=0.25, lw=0.8)
            add_panel_label(ax, f"({panel_labels[i*3 + j]})")

            if i == 0:
                ax.set_title(M, pad=10)

            if j == 0:
                ax.set_ylabel(row_labels[i])

            if i == 3:
                ax.set_xlabel(r"$(x-x_s)/\lambda_1$")

    fig.suptitle(
        "Kinetic standing-shock reference profiles across Mach number",
        y=0.985, fontsize=16
    )

    fig.subplots_adjust(
        left=0.10, right=0.98,
        bottom=0.07, top=0.93,
        wspace=0.25, hspace=0.34
    )

    outstem = Path(args.out)
    outstem.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(outstem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(outstem) + ".png", bbox_inches="tight")
    print("[done]", str(outstem) + ".pdf")
    print("[done]", str(outstem) + ".png")


if __name__ == "__main__":
    main()
