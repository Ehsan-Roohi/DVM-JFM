#!/usr/bin/env python3
"""
Exact 1D steady heat-flux moment budget from a DVM fullstate.

This diagnostic is for the next paper:
  Can Rxx be made observable through a heat-flux moment equation?

Main outputs:
  1) exact heat-flux flux:
       F_q = int v_x * (1/2 |c|^2 c_x) f dv

  2) exact algebraic decomposition:
       F_q = u q + 1/2 [ 5 rho T^2 + 7 T sigma_xx + Rxx_closure + Delta/3 ]

  3) exact heat-flux balance:
       dF_q/dx + u_x * [ u(5/2 rho T + sigma_xx) + mxxx + 11/5 qx ] + qx/Kn = 0

  4) reconstruction of observable combination:
       S = Rxx_closure + Delta/3

     from the heat-flux balance without using R-data.

Interpretation:
  If S is reconstructed well but R alone is not, then heat-flux residual observes
  the R+Delta/3 combination. That points to a minimal R26-informed neural moment system.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_key(z, candidates, required=True):
    for k in candidates:
        if k in z.files:
            return k
    if required:
        raise KeyError(f"None of {candidates} found. Available keys:\n{z.files}")
    return None


def load_velocity_grid(z):
    if all(k in z.files for k in ["vx", "vy", "vz", "w"]):
        vx = np.asarray(z["vx"], dtype=np.float64).reshape(-1)
        vy = np.asarray(z["vy"], dtype=np.float64).reshape(-1)
        vz = np.asarray(z["vz"], dtype=np.float64).reshape(-1)
        w = np.asarray(z["w"], dtype=np.float64).reshape(-1)
        return vx, vy, vz, w

    if "v" in z.files and "w" in z.files:
        v = np.asarray(z["v"], dtype=np.float64)
        w = np.asarray(z["w"], dtype=np.float64).reshape(-1)
        if v.ndim != 2:
            raise ValueError(f"v must be 2D, got shape {v.shape}")
        if v.shape[1] == 3:
            vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        elif v.shape[0] == 3:
            vx, vy, vz = v[0, :], v[1, :], v[2, :]
        else:
            raise ValueError(f"Cannot interpret v shape {v.shape}")
        return vx.reshape(-1), vy.reshape(-1), vz.reshape(-1), w

    raise KeyError(f"Could not find velocity grid. Need vx,vy,vz,w or v,w. Keys:\n{z.files}")


def get_f_view(z, nx, nv):
    kf = pick_key(z, ["f", "F", "states", "dist", "distribution"])
    F = np.asarray(z[kf])
    if F.ndim != 2:
        raise ValueError(f"Distribution {kf} must be 2D, got {F.shape}")

    if F.shape == (nx, nv):
        return kf, "xv", F
    if F.shape == (nv, nx):
        return kf, "vx", F

    raise ValueError(f"Cannot match f shape {F.shape} with nx={nx}, nv={nv}")


def f_chunk(F, orient, i0, i1):
    if orient == "xv":
        return np.asarray(F[i0:i1, :], dtype=np.float64)
    return np.asarray(F[:, i0:i1].T, dtype=np.float64)


def ddx(y, x):
    return np.gradient(y, x, edge_order=2)


def smooth(y, passes=0):
    if passes <= 0:
        return np.asarray(y, dtype=np.float64)
    out = np.asarray(y, dtype=np.float64).copy()
    ker = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    ker /= ker.sum()
    for _ in range(passes):
        pad = np.pad(out, (2, 2), mode="edge")
        out = np.convolve(pad, ker, mode="valid")
    return out


def cumtrapz(y, x):
    out = np.zeros_like(y, dtype=np.float64)
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def rel_l2(a, b, x, mask=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dxw = np.gradient(x)
    if mask is None:
        mask = np.ones_like(b, dtype=bool)
    num = np.sqrt(np.sum(((a - b) ** 2) * dxw * mask))
    den = np.sqrt(np.sum((b ** 2) * dxw * mask)) + 1e-300
    return num / den


def corrcoef(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-300 or np.std(b) < 1e-300:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="DVM fullstate npz containing f,v,w,x,rho,ux,T")
    ap.add_argument("--case", default=None)
    ap.add_argument("--outdir", default="appendix_validation_dvm/exact_heatflux_budget")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--smooth-deriv", type=int, default=1)
    ap.add_argument("--kn", type=float, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = Path("figures/exact_heatflux_budget")
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.ref, allow_pickle=True)
    print("[file]", args.ref)
    print("[keys]", z.files)

    kx = pick_key(z, ["x_mfp", "x", "x_scaled"])
    x = np.asarray(z[kx], dtype=np.float64).reshape(-1)
    nx = len(x)

    rho_key = pick_key(z, ["rho"])
    ux_key = pick_key(z, ["ux", "u", "u_x"])
    T_key = pick_key(z, ["T", "temp"])

    rho = np.asarray(z[rho_key], dtype=np.float64).reshape(-1)
    u = np.asarray(z[ux_key], dtype=np.float64).reshape(-1)
    T = np.asarray(z[T_key], dtype=np.float64).reshape(-1)

    vx, vy, vz, w = load_velocity_grid(z)
    nv = len(w)

    kf, orient, F = get_f_view(z, nx, nv)
    print(f"[f] key={kf}, orient={orient}, shape={F.shape}")
    print(f"[v] Nv={nv}, xkey={kx}, nx={nx}")

    if args.kn is not None:
        kn = args.kn
    elif "kn_eff" in z.files:
        kn = float(np.asarray(z["kn_eff"]).reshape(-1)[0])
    elif "kn" in z.files:
        kn = float(np.asarray(z["kn"]).reshape(-1)[0])
    else:
        kn = 1.0 / 80.0
        print("[warning] no kn_eff/kn found; using 1/80")
    print("[kn]", kn)

    idx = np.argsort(x)
    if not np.all(idx == np.arange(nx)):
        print("[sort] sorting x and fields")
        x = x[idx]
        rho = rho[idx]
        u = u[idx]
        T = T[idx]
        # if F orientation has x dimension, reorder below by making a sorted copy
        if orient == "xv":
            F = F[idx, :]
        else:
            F = F[:, idx]

    # Moment arrays computed from f.
    rho_f = np.zeros(nx)
    u_f = np.zeros(nx)
    T_f = np.zeros(nx)
    q = np.zeros(nx)
    sig = np.zeros(nx)
    m = np.zeros(nx)
    R = np.zeros(nx)
    Delta = np.zeros(nx)
    Fq_direct = np.zeros(nx)

    ww = w[None, :]
    vx0 = vx[None, :]
    vy0 = vy[None, :]
    vz0 = vz[None, :]

    for i0 in range(0, nx, args.chunk):
        i1 = min(nx, i0 + args.chunk)
        ff = f_chunk(F, orient, i0, i1)
        uu = u[i0:i1, None]
        TT = T[i0:i1]
        rr = rho[i0:i1]

        cx = vx0 - uu
        cy = vy0
        cz = vz0
        c2 = cx * cx + cy * cy + cz * cz

        fw = ff * ww

        rho_f[i0:i1] = np.sum(fw, axis=1)
        momx = np.sum(vx0 * fw, axis=1)
        u_f[i0:i1] = momx / (rho_f[i0:i1] + 1e-300)

        C2raw = np.sum(c2 * fw, axis=1)
        T_f[i0:i1] = C2raw / (3.0 * rho_f[i0:i1] + 1e-300)

        q[i0:i1] = 0.5 * np.sum(c2 * cx * fw, axis=1)
        sig[i0:i1] = np.sum((cx * cx - c2 / 3.0) * fw, axis=1)

        # Closure-level third moment. Maxwellian contribution is analytically zero.
        m[i0:i1] = np.sum((cx**3 - (3.0/5.0) * cx * c2) * fw, axis=1)

        # Fourth-order scalar excess.
        C4 = np.sum((c2 * c2) * fw, axis=1)
        Delta[i0:i1] = C4 - 15.0 * rr * TT**2

        # R closure.
        B = np.sum(c2 * (cx * cx - c2 / 3.0) * fw, axis=1)
        R[i0:i1] = B - 7.0 * TT * sig[i0:i1]

        # Direct heat-flux-moment flux:
        # Fq = int v_x * 0.5 c^2 c_x f dv
        Fq_direct[i0:i1] = np.sum(vx0 * 0.5 * c2 * cx * fw, axis=1)

        print(f"[chunk] {i0}:{i1}")

    p = rho * T

    # Algebraic decomposition:
    # Fq = u q + 1/2 [5 rho T^2 + 7 T sigma + R + Delta/3]
    S_ref = R + Delta / 3.0
    Fq_decomp = u * q + 0.5 * (5.0 * rho * T**2 + 7.0 * T * sig + S_ref)

    # Smooth only for derivatives.
    us = smooth(u, args.smooth_deriv)
    qs = smooth(q, args.smooth_deriv)
    sigs = smooth(sig, args.smooth_deriv)
    ms = smooth(m, args.smooth_deriv)
    Fqs = smooth(Fq_direct, args.smooth_deriv)

    ux = ddx(us, x)
    Fq_x_direct = ddx(Fqs, x)

    # Exact central heat-flux balance derived from multiplying steady BGK by
    # Hq = 0.5 c^2 c_x:
    # dFq/dx + ux * [ u(5/2 rhoT + sigma) + m + 11/5 q ] + q/Kn = 0
    geom = ux * (u * (2.5 * p + sig) + m + (11.0/5.0) * q)
    heat_residual = Fq_x_direct + geom + q / kn

    # Reconstruction of Fq from the heat-flux balance, using only lower fields and m.
    Fq_prime_balance = -geom - q / kn

    # Left equilibrium constant: in a Maxwellian plateau, q=sigma=m=R=Delta=0,
    # so Fq = 1/2 * 5 rho T^2.
    nedge = max(5, nx // 30)
    Fq_left_const = float(np.mean(2.5 * rho[:nedge] * T[:nedge]**2))
    Fq_rec = Fq_left_const + cumtrapz(Fq_prime_balance, x)

    # Observable combination from heat flux:
    # S = R + Delta/3 = 2(Fq - u q) - 5 rho T^2 - 7 T sigma.
    S_rec = 2.0 * (Fq_rec - u * q) - 5.0 * rho * T**2 - 7.0 * T * sig

    # R if Delta is known. This is NOT data-free for R26, but diagnoses whether
    # missing Delta is the obstruction.
    R_rec_with_Delta = S_rec - Delta / 3.0

    # R if Delta is ignored. This is the minimal R-only attempt.
    R_rec_Delta0 = S_rec.copy()

    # Active masks.
    mask_R = np.abs(R) > 0.05 * (np.max(np.abs(R)) + 1e-300)
    mask_S = np.abs(S_ref) > 0.05 * (np.max(np.abs(S_ref)) + 1e-300)
    mask_q = np.abs(q) > 0.05 * (np.max(np.abs(q)) + 1e-300)

    case = args.case or Path(args.ref).stem

    summary = pd.DataFrame([{
        "case": case,
        "file": args.ref,
        "kn": kn,
        "nx": nx,
        "nv": nv,
        "smooth_deriv": args.smooth_deriv,

        "rho_rel_diff_file_vs_f": rel_l2(rho_f, rho, x),
        "u_rel_diff_file_vs_f": rel_l2(u_f, u, x),
        "T_rel_diff_file_vs_f": rel_l2(T_f, T, x),

        "Fq_decomp_relL2": rel_l2(Fq_decomp, Fq_direct, x),
        "Fq_decomp_corr": corrcoef(Fq_decomp, Fq_direct),

        "heat_residual_rel_to_q_over_kn_full": np.sqrt(np.sum(heat_residual**2 * np.gradient(x))) /
                                                (np.sqrt(np.sum((q/kn)**2 * np.gradient(x))) + 1e-300),
        "heat_residual_rel_to_q_over_kn_active_q": np.sqrt(np.sum((heat_residual[mask_q])**2 * np.gradient(x)[mask_q])) /
                                                        (np.sqrt(np.sum((q[mask_q]/kn)**2 * np.gradient(x)[mask_q])) + 1e-300),

        "S_R_plus_Delta3_relL2_full": rel_l2(S_rec, S_ref, x),
        "S_R_plus_Delta3_relL2_active": rel_l2(S_rec, S_ref, x, mask_S),
        "S_R_plus_Delta3_corr": corrcoef(S_rec, S_ref),

        "R_rec_with_Delta_relL2_full": rel_l2(R_rec_with_Delta, R, x),
        "R_rec_with_Delta_relL2_active": rel_l2(R_rec_with_Delta, R, x, mask_R),
        "R_rec_with_Delta_corr": corrcoef(R_rec_with_Delta, R),

        "R_rec_Delta0_relL2_full": rel_l2(R_rec_Delta0, R, x),
        "R_rec_Delta0_relL2_active": rel_l2(R_rec_Delta0, R, x, mask_R),
        "R_rec_Delta0_corr": corrcoef(R_rec_Delta0, R),

        "R_peak": np.max(np.abs(R)),
        "Delta_peak": np.max(np.abs(Delta)),
        "S_peak": np.max(np.abs(S_ref)),
        "R_over_S_peak": np.max(np.abs(R)) / (np.max(np.abs(S_ref)) + 1e-300),
        "Delta3_over_S_peak": np.max(np.abs(Delta/3.0)) / (np.max(np.abs(S_ref)) + 1e-300),
    }])

    profiles = pd.DataFrame({
        "x": x,
        "rho_file": rho,
        "rho_from_f": rho_f,
        "ux_file": u,
        "ux_from_f": u_f,
        "T_file": T,
        "T_from_f": T_f,
        "qx": q,
        "sigma_xx": sig,
        "mxxx": m,
        "Rxx_closure": R,
        "Delta": Delta,
        "S_R_plus_Delta3_ref": S_ref,
        "Fq_direct": Fq_direct,
        "Fq_decomp": Fq_decomp,
        "Fq_rec_from_balance": Fq_rec,
        "heat_residual": heat_residual,
        "S_R_plus_Delta3_rec": S_rec,
        "R_rec_with_Delta": R_rec_with_Delta,
        "R_rec_Delta0": R_rec_Delta0,
        "ux_grad": ux,
        "geom_term": geom,
        "q_over_kn": q / kn,
    })

    summary_csv = outdir / f"{case}_exact_heatflux_budget_summary.csv"
    profiles_csv = outdir / f"{case}_exact_heatflux_budget_profiles.csv"
    summary.to_csv(summary_csv, index=False)
    profiles.to_csv(profiles_csv, index=False)

    def savefig(name):
        png = figdir / f"{case}_{name}.png"
        pdf = figdir / f"{case}_{name}.pdf"
        plt.tight_layout()
        plt.savefig(png, dpi=300)
        plt.savefig(pdf)
        print("[plot]", png)
        print("[plot]", pdf)

    plt.figure(figsize=(8, 5))
    plt.plot(x, Fq_direct, "k-", lw=2.5, label="direct $F_q$")
    plt.plot(x, Fq_decomp, "--", lw=2.0, label="algebraic decomposition")
    plt.plot(x, Fq_rec, lw=1.8, label="from heat-flux balance")
    plt.xlabel(r"$x/\lambda_1$")
    plt.ylabel(r"$F_q$")
    plt.title(f"Exact heat-flux flux budget: {case}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("Fq_budget")

    plt.figure(figsize=(8, 5))
    plt.plot(x, S_ref, "k--", lw=2.5, label=r"DVM $R_{xx}^{cl}+\Delta/3$")
    plt.plot(x, S_rec, lw=2.0, label="reconstructed from heat-flux balance")
    plt.axhline(0, color="0.4", lw=0.8)
    plt.xlabel(r"$x/\lambda_1$")
    plt.ylabel(r"$R_{xx}^{cl}+\Delta/3$")
    plt.title(f"Observable fourth-order combination: {case}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("S_R_plus_Delta3_reconstruction")

    plt.figure(figsize=(8, 5))
    plt.plot(x, R, "k--", lw=2.5, label=r"DVM $R_{xx}^{cl}$")
    plt.plot(x, R_rec_Delta0, lw=1.8, label=r"reconstructed with $\Delta=0$")
    plt.plot(x, R_rec_with_Delta, lw=1.8, label=r"reconstructed using DVM $\Delta$")
    plt.axhline(0, color="0.4", lw=0.8)
    plt.xlabel(r"$x/\lambda_1$")
    plt.ylabel(r"$R_{xx}^{cl}$")
    plt.title(f"Can heat-flux balance recover R alone? {case}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("R_reconstruction")

    plt.figure(figsize=(8, 5))
    plt.plot(x, Fq_x_direct, lw=2.0, label=r"$\partial_x F_q$")
    plt.plot(x, geom, lw=2.0, label="central-moment geometry term")
    plt.plot(x, q/kn, lw=2.0, label=r"$q_x/\mathrm{Kn}$")
    plt.plot(x, heat_residual, "k--", lw=2.0, label="sum residual")
    plt.axhline(0, color="0.4", lw=0.8)
    plt.xlabel(r"$x/\lambda_1$")
    plt.ylabel("budget terms")
    plt.title(f"Heat-flux moment balance residual: {case}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("heatflux_balance_terms")

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n[saved]")
    print(summary_csv)
    print(profiles_csv)


if __name__ == "__main__":
    main()
