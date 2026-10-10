#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def shock_center_from_density(x, n, edge_frac=0.12):
    x = np.asarray(x, float)
    n = np.asarray(n, float)

    m = len(x)
    ne = max(5, int(edge_frac*m))

    nL = np.median(n[:ne])
    nR = np.median(n[-ne:])
    nmid = 0.5*(nL+nR)

    y = n - nmid

    # find first crossing from low to high
    idx = np.where(y[:-1]*y[1:] <= 0)[0]
    if len(idx) == 0:
        # fallback: max density gradient
        dn = np.gradient(n, x)
        return float(x[np.argmax(np.abs(dn))]), nL, nR

    # choose crossing closest to max gradient
    dn = np.gradient(n, x)
    ig = int(np.argmax(np.abs(dn)))
    k = int(idx[np.argmin(np.abs(idx - ig))])

    x0, x1 = x[k], x[k+1]
    y0, y1 = y[k], y[k+1]
    if abs(y1-y0) < 1e-300:
        xs = 0.5*(x0+x1)
    else:
        xs = x0 - y0*(x1-x0)/(y1-y0)

    return float(xs), nL, nR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge-frac", type=float, default=0.12)
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found for pattern: {args.pattern}")

    print("[files]")
    for f in files:
        print(" ", f)

    dfs = []
    centers = []

    for f in files:
        df = pd.read_csv(f)
        if "x" not in df or "n" not in df:
            raise KeyError(f"{f} must contain x and n columns")

        x = df["x"].to_numpy(float)
        n = df["n"].to_numpy(float)

        xs, nL, nR = shock_center_from_density(x, n, args.edge_frac)
        centers.append((str(f), xs, nL, nR))

        df = df.copy()
        df["x_centered"] = df["x"] - xs
        dfs.append(df)

    print("\n[shock centers]")
    for f, xs, nL, nR in centers:
        print(f"{f:70s} xs={xs:+.6f} nL={nL:.6g} nR={nR:.6g}")

    # common centered grid
    dxs = []
    xmin_list = []
    xmax_list = []
    for df in dfs:
        x = df["x_centered"].to_numpy(float)
        dxs.append(np.median(np.diff(x)))
        xmin_list.append(x.min())
        xmax_list.append(x.max())

    dx = float(np.median(dxs))
    xmin = max(xmin_list)
    xmax = min(xmax_list)

    # snap to grid
    i0 = int(np.ceil(xmin/dx))
    i1 = int(np.floor(xmax/dx))
    x_common = np.arange(i0, i1+1)*dx

    print(f"\n[common grid] nx={len(x_common)} dx={dx:.6g} xmin={x_common[0]:.6g} xmax={x_common[-1]:.6g}")

    cols = list(dfs[0].columns)
    numeric_cols = []
    for c in cols:
        if c in ["x", "x_centered"]:
            continue
        if np.issubdtype(dfs[0][c].dtype, np.number):
            numeric_cols.append(c)

    out = pd.DataFrame({"x": x_common})

    for c in numeric_cols:
        vals = []
        for df in dfs:
            xi = df["x_centered"].to_numpy(float)
            yi = df[c].to_numpy(float)
            vals.append(np.interp(x_common, xi, yi))
        vals = np.asarray(vals)

        out[c] = vals.mean(axis=0)
        out[c + "_std"] = vals.std(axis=0, ddof=1) if len(vals) > 1 else 0.0

    out["n_runs"] = len(dfs)
    out["shock_center_mean"] = np.mean([c[1] for c in centers])
    out["shock_center_std"] = np.std([c[1] for c in centers], ddof=1) if len(centers) > 1 else 0.0

    out.to_csv(args.out, index=False)
    print("[done]", args.out)


if __name__ == "__main__":
    main()
