#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_file(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError("Could not find any of:\n" + "\n".join(map(str, paths)))


def pick(df, names):
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


def active_mask(y, thresh=0.02):
    peak = np.max(np.abs(y))
    if peak < 1e-30:
        return np.ones_like(y, dtype=bool)
    m = np.abs(y) >= thresh*peak
    if m.sum() < 10:
        return np.ones_like(y, dtype=bool)
    return m


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


def nice_ylim(*ys, pad=0.08):
    vals = np.concatenate([np.asarray(y).ravel() for y in ys])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    if abs(hi-lo) < 1e-14:
        return lo-1, hi+1
    d = hi-lo
    return lo-pad*d, hi+pad*d


def load_case(M):
    # Heat-flux budget profiles.
    hfile = find_file([
        f"appendix_validation_dvm/exact_heatflux_budget/{M}_exact_budget_xmfp_exact_heatflux_budget_profiles.csv",
        f"appendix_validation_dvm/exact_heatflux_budget/{M}_exact_budget_exact_heatflux_budget_profiles.csv",
    ])
    h = pd.read_csv(hfile)

    x, _ = pick(h, ["x"])
    R, _ = pick(h, ["Rxx_closure", "R", "Rxx", "R_ref"])
    D, _ = pick(h, ["Delta", "Delta_true", "Delta_ref"])
    S, sname = pick(h, ["S_R_plus_Delta3_rec", "S_R_plus_Delta3_ref", "S", "S_true", "S_target"])

    R = plateau_zero(R)
    D = plateau_zero(D)
    S = plateau_zero(S)

    # Delta-budget profiles.
    # Prefer the clean M_delta_budget case; fall back to smooth7.
    dfile = find_file([
        f"figures/week2_jfm_upgrade_delta_budget/{M}_delta_budget/{M}_delta_budget.profiles.csv",
        f"figures/week2_jfm_upgrade_delta_budget/{M}_delta_budget_smooth7/{M}_delta_budget_smooth7.profiles.csv",
    ])
    db = pd.read_csv(dfile)
    xd, _ = pick(db, ["x"])

    D_budget, _ = pick(db, ["Delta_rec_from_budget", "Delta_budget", "Delta_rec"])
    D_budget = plateau_zero(D_budget)

    # Interpolate budget Delta onto heat-flux grid if needed.
    if len(xd) != len(x) or np.max(np.abs(xd-x[:len(xd)])) if len(xd)==len(x) else True:
        D_budget_i = np.interp(x, xd, D_budget)
    else:
        D_budget_i = D_budget

    # If direct Delta in delta-budget file differs slightly, use heat-flux file D as reference.
    R_no_delta = S.copy()
    R_full_delta = S - D/3.0
    R_budget_delta = S - D_budget_i/3.0

    mask = active_mask(S, 0.02)

    metrics = {
        "Mach": M,
        "heatflux_file": str(hfile),
        "delta_budget_file": str(dfile),
        "S_source": sname,

        "Delta_budget_relL2_active": rel_l2(D_budget_i, D, mask),
        "Delta_budget_corr_active": corr(D_budget_i, D, mask),

        "R_delta0_relL2_active": rel_l2(R_no_delta, R, mask),
        "R_fullDelta_relL2_active": rel_l2(R_full_delta, R, mask),
        "R_budgetDelta_relL2_active": rel_l2(R_budget_delta, R, mask),
        "R_budgetDelta_corr_active": corr(R_budget_delta, R, mask),

        "R_delta0_relL2_full": rel_l2(R_no_delta, R),
        "R_fullDelta_relL2_full": rel_l2(R_full_delta, R),
        "R_budgetDelta_relL2_full": rel_l2(R_budget_delta, R),
    }

    return {
        "M": M, "x": x,
        "R": R, "D": D, "S": S, "D_budget": D_budget_i,
        "R_no_delta": R_no_delta,
        "R_full_delta": R_full_delta,
        "R_budget_delta": R_budget_delta,
        "mask": mask,
        "metrics": metrics,
    }


def main():
    outdir = Path("figures/week2_jfm_upgrade_two_budget")
    outdir.mkdir(parents=True, exist_ok=True)

    cases = [load_case(M) for M in ["M2", "M3", "M5"]]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.4,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.5), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.92, wspace=0.25, hspace=0.34)

    titles = [
        r"Scalar-excess budget",
        r"Heat-flux channel alone",
        r"Two-budget closure recovery",
    ]
    for j, t in enumerate(titles):
        axs[0, j].set_title(t, pad=7)

    letters = list("abcdefghi")

    rows = []

    for i, c in enumerate(cases):
        M = c["M"]
        x = c["x"]
        R = c["R"]
        D = c["D"]
        S = c["S"]
        D_budget = c["D_budget"]
        R_no = c["R_no_delta"]
        R_full = c["R_full_delta"]
        R_budget = c["R_budget_delta"]
        met = c["metrics"]

        eD = 100*met["Delta_budget_relL2_active"]
        eR0 = 100*met["R_delta0_relL2_active"]
        eRfull = 100*met["R_fullDelta_relL2_active"]
        eRbud = 100*met["R_budgetDelta_relL2_active"]
        rhoR = met["R_budgetDelta_corr_active"]

        rows.append({
            "Mach": M,
            r"$\epsilon_\Delta$ budget (\%)": eD,
            r"$\rho_\Delta$": met["Delta_budget_corr_active"],
            r"$\epsilon_R(\Delta=0)$ (\%)": eR0,
            r"$\epsilon_R(\Delta_{\rm full})$ (\%)": eRfull,
            r"$\epsilon_R(\Delta_{\rm budget})$ (\%)": eRbud,
            r"$\rho_R(\Delta_{\rm budget})$": rhoR,
        })

        # Col 1: Delta budget.
        ax = axs[i, 0]
        ax.plot(x, D, "k-", label=r"DVM $\Delta$")
        ax.plot(x, D_budget, "r--", label=r"budget $\Delta$")
        ax.set_ylim(*nice_ylim(D, D_budget))
        ax.grid(True, alpha=0.22, lw=0.5)
        ax.text(0.03, 0.94, f"({letters[3*i]})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        ax.text(0.97, 0.94, rf"$\epsilon_\Delta={eD:.2f}\%$",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="0.85", pad=2.0))
        if i == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=7.2)

        # Col 2: Heat-flux only.
        ax = axs[i, 1]
        ax.plot(x, R, "k-", label=r"DVM $R^{cl}_{xx}$")
        ax.plot(x, R_no, "r--", label=r"$S$, $\Delta=0$")
        ax.set_ylim(*nice_ylim(R, R_no))
        ax.grid(True, alpha=0.22, lw=0.5)
        ax.text(0.03, 0.94, f"({letters[3*i+1]})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        ax.text(0.97, 0.94, rf"$\epsilon_R={eR0:.1f}\%$",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="0.85", pad=2.0))
        if i == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=7.2)

        # Col 3: Two-budget recovery.
        ax = axs[i, 2]
        ax.plot(x, R, "k-", label=r"DVM $R^{cl}_{xx}$")
        ax.plot(x, R_full, color="0.55", ls=":", label=r"full $\Delta$")
        ax.plot(x, R_budget, "C2--", label=r"budget $\Delta$")
        ax.set_ylim(*nice_ylim(R, R_full, R_budget))
        ax.grid(True, alpha=0.22, lw=0.5)
        ax.text(0.03, 0.94, f"({letters[3*i+2]})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        ax.text(0.97, 0.94, rf"$\epsilon_R={eRbud:.2f}\%$"+"\n"+rf"$\rho_R={rhoR:.5f}$",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="0.85", pad=2.0))
        if i == 0:
            ax.legend(loc="lower left", frameon=True, fontsize=7.2)

        # Row labels.
        bbox = axs[i, 0].get_position()
        ymid = 0.5*(bbox.y0 + bbox.y1)
        fig.text(0.025, ymid, M, rotation=90, ha="center", va="center",
                 fontsize=10.2, fontweight="bold")
        fig.text(0.055, ymid, "Moment value", rotation=90, ha="center", va="center",
                 fontsize=8.5)

        if i == 2:
            for j in range(3):
                axs[i, j].set_xlabel(r"$(x-x_s)/\lambda_1$")

    outstem = outdir / "two_budget_recovery_jfm"
    fig.savefig(str(outstem)+".pdf", bbox_inches="tight")
    fig.savefig(str(outstem)+".png", dpi=500, bbox_inches="tight")
    plt.close(fig)

    tab = pd.DataFrame(rows)
    tab.to_csv(outdir / "two_budget_recovery_summary.csv", index=False)

    tex = tab.to_latex(
        index=False,
        escape=False,
        float_format="%.3f",
        caption=(
            r"Two-budget recovery of \(R^{\mathrm{cl}}_{xx}\). "
            r"The heat-flux budget gives \(S=R^{\mathrm{cl}}_{xx}+\Delta/3\), "
            r"while the scalar-excess budget gives \(\Delta\)."
        ),
        label="tab:two_budget_recovery",
    )
    (outdir / "two_budget_recovery_summary.tex").write_text(tex)

    print("[done]", outstem.with_suffix(".pdf"))
    print("[done]", outstem.with_suffix(".png"))
    print("[done]", outdir / "two_budget_recovery_summary.csv")
    print("[done]", outdir / "two_budget_recovery_summary.tex")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
