from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")

CASES = [
    {
        "mach": "M2",
        "csv": ROOT / "figures/esfp_scalar_source/M2_ESFP_source_from_BGK_DVM/M2_ESFP_source_from_BGK_DVM.csv",
        "metrics": ROOT / "figures/esfp_scalar_source/M2_ESFP_source_from_BGK_DVM/M2_ESFP_source_from_BGK_DVM_metrics.csv",
    },
    {
        "mach": "M3",
        "csv": ROOT / "figures/esfp_scalar_source/M3_ESFP_source_from_BGK_DVM/M3_ESFP_source_from_BGK_DVM.csv",
        "metrics": ROOT / "figures/esfp_scalar_source/M3_ESFP_source_from_BGK_DVM/M3_ESFP_source_from_BGK_DVM_metrics.csv",
    },
    {
        "mach": "M5",
        "csv": ROOT / "figures/esfp_scalar_source/M5_ESFP_source_from_BGK_DVM/M5_ESFP_source_from_BGK_DVM.csv",
        "metrics": ROOT / "figures/esfp_scalar_source/M5_ESFP_source_from_BGK_DVM/M5_ESFP_source_from_BGK_DVM_metrics.csv",
    },
]

def get_metric(met, keys, default=np.nan):
    for k in keys:
        if k in met.index:
            return float(met[k])
    return default

def check_columns(df, required, fname):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{fname} is missing columns: {missing}\nAvailable: {list(df.columns)}")

outdir = ROOT / "figures/esfp_scalar_source/publication_figure"
outdir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.3,
})

fig, axs = plt.subplots(1, 3, figsize=(15.8, 5.2), constrained_layout=False)
fig.subplots_adjust(left=0.065, right=0.995, bottom=0.22, top=0.76, wspace=0.24)

legend_handles = None
legend_labels = None

for ax, case in zip(axs, CASES):
    mach = case["mach"]
    csv_path = case["csv"]
    met_path = case["metrics"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not met_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {met_path}")

    df = pd.read_csv(csv_path)
    met = pd.read_csv(met_path).iloc[0]

    required = ["x", "Q_BGK_like_scaled", "Q_ESFP_scaled", "Q_ESBGK_scaled"]
    check_columns(df, required, csv_path)

    x = df["x"].to_numpy()
    q_bgk = df["Q_BGK_like_scaled"].to_numpy()
    q_esfp = df["Q_ESFP_scaled"].to_numpy()
    q_esbgk = df["Q_ESBGK_scaled"].to_numpy()

    l1, = ax.plot(x, q_bgk, color="k", lw=2.5, label=r"BGK/Shakhov: $-\Delta$")
    l2, = ax.plot(x, q_esbgk, color="tab:green", lw=2.4, ls=":", label=r"ES-BGK source")
    l3, = ax.plot(x, q_esfp, color="red", lw=2.4, ls="--", label=r"ES-FP source")

    if legend_handles is None:
        legend_handles = [l1, l2, l3]
        legend_labels = [h.get_label() for h in legend_handles]

    esfp_corr = get_metric(met, ["ESFP_corr_over_Delta_active_direct", "ESFP_corr_over_Delta_active"])
    esbgk_corr = get_metric(met, ["ESBGK_corr_over_Delta_active"])
    corr_esfp = get_metric(met, ["ESFP_scaled_vs_BGK_corr_active", "Q_ESFP_vs_BGK_corr_active"])

    ax.set_title(mach, pad=8)
    ax.set_xlabel(r"$(x-x_s)/\lambda_1$")
    ax.grid(True, alpha=0.28)

    if ax is axs[0]:
        ax.set_ylabel(r"scaled scalar-excess source")

    txt = (
        rf"ES-FP corr. $= {esfp_corr:.3f}$" "\n"
        rf"ES-BGK corr. $= {esbgk_corr:.3f}$" "\n"
        rf"$\rho$ with $-\Delta$ $= {corr_esfp:.3f}$"
    )

    ax.text(
        0.035, 0.055, txt,
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="0.7")
    )

    ax.text(
        0.50, -0.25, f"({chr(ord('a') + CASES.index(case))})",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=13
    )

fig.legend(
    legend_handles, legend_labels,
    loc="upper center",
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, 0.985),
    borderaxespad=0.0
)

pdf_path = outdir / "esfp_esbgk_scaled_source_M2_M3_M5.pdf"
png_path = outdir / "esfp_esbgk_scaled_source_M2_M3_M5.png"

fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("[done]", pdf_path)
print("[done]", png_path)
