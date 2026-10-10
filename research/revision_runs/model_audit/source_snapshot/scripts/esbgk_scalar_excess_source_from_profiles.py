#!/usr/bin/env python3
"""
Post-process a 1D normal-shock moment profile and evaluate the ES-BGK
scalar-excess target/source implied by an anisotropic Gaussian target.

For a trace-free normal stress tensor in a 1D normal shock,
    sigma = diag(sigma_xx, -sigma_xx/2, -sigma_xx/2),
so tr(sigma^2)=3 sigma_xx^2/2.
If Theta = T I + alpha_ES sigma/rho, then
    Delta_G^ES = 2 alpha_ES^2 tr(sigma^2)/rho
               = 3 alpha_ES^2 sigma_xx^2 / rho.
The ES-BGK scalar-excess collision source is
    Q_Delta^ES = (Delta_G^ES - Delta)/Kn.

This script does not solve an ES-BGK shock.  It is a source-level diagnostic
that can be applied to any DVM profile containing rho, sigma_xx and Delta.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pick(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Available columns: {list(df.columns)}")


def plateau_zero(y: np.ndarray, frac: float = 0.08) -> np.ndarray:
    y = np.asarray(y, float).copy()
    n = len(y)
    ne = max(5, int(frac*n))
    idx = np.r_[np.arange(ne), np.arange(n-ne, n)]
    return y - np.mean(y[idx])


def rel_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    den = np.linalg.norm(b[mask])
    return float(np.linalg.norm(a[mask]-b[mask]) / max(den, 1e-300))


def corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = a[mask] - np.mean(a[mask])
    bb = b[mask] - np.mean(b[mask])
    den = np.linalg.norm(aa)*np.linalg.norm(bb)
    return float(np.dot(aa, bb)/max(den, 1e-300))


def active_mask(y: np.ndarray, thresh: float) -> np.ndarray:
    p = np.max(np.abs(y))
    if p < 1e-300:
        return np.ones_like(y, dtype=bool)
    m = np.abs(y) > thresh*p
    if m.sum() < 10:
        m[:] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True, help="Input profile CSV containing rho, sigma_xx and Delta")
    ap.add_argument("--outdir", default="figures/esbgk_scalar_source", help="Output directory")
    ap.add_argument("--case", default=None, help="Case name; defaults to input stem")
    ap.add_argument("--Pr", type=float, default=2.0/3.0)
    ap.add_argument("--Kn", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="ES-BGK stress coefficient alpha_ES. Default: 1-1/Pr.")
    ap.add_argument("--active-thresh", type=float, default=0.02)
    args = ap.parse_args()

    inp = Path(args.profiles)
    case = args.case or inp.stem
    outdir = Path(args.outdir) / case
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    xcol = pick(df, ["x", "x_mfp", "x_center", "x_raw"])
    rcol = pick(df, ["rho", "rho_file"])
    scol = pick(df, ["sigma_xx", "sigma", "sig"])
    dcol = pick(df, ["Delta", "Delta_ref", "Delta_true"])

    x = df[xcol].to_numpy(float)
    rho = df[rcol].to_numpy(float)
    sig = df[scol].to_numpy(float)
    Delta = plateau_zero(df[dcol].to_numpy(float))

    alpha = args.alpha if args.alpha is not None else (1.0 - 1.0/args.Pr)

    # 1D trace-free stress tensor: tr(sigma^2)=3 sigma_xx^2/2.
    tr_sigma2 = 1.5 * sig*sig
    Delta_G = 2.0 * alpha*alpha * tr_sigma2 / rho
    Q_delta_ES = (Delta_G - Delta) / args.Kn
    Q_delta_BGK_or_Shakhov = -Delta / args.Kn

    mask = active_mask(Delta, args.active_thresh)

    metrics = {
        "case": case,
        "Pr": args.Pr,
        "Kn": args.Kn,
        "alpha_ES": alpha,
        "DeltaG_over_Delta_active": float(np.linalg.norm(Delta_G[mask]) / max(np.linalg.norm(Delta[mask]), 1e-300)),
        "Q_ES_vs_BGK_relL2_active": rel_l2(Q_delta_ES, Q_delta_BGK_or_Shakhov, mask),
        "Q_ES_vs_BGK_corr_active": corr(Q_delta_ES, Q_delta_BGK_or_Shakhov, mask),
        "Delta_peak": float(np.max(np.abs(Delta))),
        "DeltaG_peak": float(np.max(np.abs(Delta_G))),
    }

    out = pd.DataFrame({
        "x": x,
        "rho": rho,
        "sigma_xx": sig,
        "Delta": Delta,
        "tr_sigma2_1D": tr_sigma2,
        "Delta_G_ES": Delta_G,
        "Q_delta_ES": Q_delta_ES,
        "Q_delta_BGK_or_Shakhov": Q_delta_BGK_or_Shakhov,
        "active_mask": mask.astype(int),
    })
    out.to_csv(outdir / f"{case}_esbgk_scalar_source.csv", index=False)
    pd.DataFrame([metrics]).to_csv(outdir / f"{case}_esbgk_scalar_source_metrics.csv", index=False)

    print("=== ES-BGK scalar-excess source diagnostic ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("[done]", outdir / f"{case}_esbgk_scalar_source.csv")
    print("[done]", outdir / f"{case}_esbgk_scalar_source_metrics.csv")


if __name__ == "__main__":
    main()
