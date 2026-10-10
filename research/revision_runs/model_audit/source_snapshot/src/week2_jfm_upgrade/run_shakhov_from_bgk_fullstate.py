#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch


def key(z, names, required=True):
    for n in names:
        if n in z.files:
            return n
    if required:
        raise KeyError(f"Missing {names}; available={z.files}")
    return None


def orient_f(f, nx, nv):
    f = np.asarray(f)
    if f.shape == (nx, nv):
        return f
    if f.shape == (nv, nx):
        return f.T
    if f.ndim == 2:
        if f.shape[0] == nx:
            return f
        if f.shape[1] == nx:
            return f.T
    raise ValueError(f"Cannot orient f shape={f.shape}, nx={nx}, nv={nv}")


def load_velocity_grid(z):
    vk = key(z, ["v", "vel", "velocity", "velocities"], required=False)
    if vk is not None:
        v = np.asarray(z[vk])
        if v.ndim == 2 and v.shape[1] >= 3:
            vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        elif v.ndim == 2 and v.shape[0] >= 3:
            vx, vy, vz = v[0], v[1], v[2]
        else:
            raise ValueError(f"Cannot parse velocity array {vk}, shape={v.shape}")
    else:
        vx = np.asarray(z[key(z, ["vx", "v_x", "Vx"])])
        vy = np.asarray(z[key(z, ["vy", "v_y", "Vy"])])
        vz = np.asarray(z[key(z, ["vz", "v_z", "Vz"])])

    w = np.asarray(z[key(z, ["w", "weights", "weight", "vw"])])

    vx = vx.ravel().astype(np.float32)
    vy = vy.ravel().astype(np.float32)
    vz = vz.ravel().astype(np.float32)
    w = w.ravel().astype(np.float32)

    if not (len(vx) == len(vy) == len(vz) == len(w)):
        raise ValueError("velocity/weight length mismatch")

    return vx, vy, vz, w


def torch_moments(f, vx, vy, vz, w):
    rho = torch.sum(f * w[None, :], dim=1).clamp_min(1e-30)
    momx = torch.sum(f * (w * vx)[None, :], dim=1)
    ux = momx / rho

    v2 = vx*vx + vy*vy + vz*vz
    E2 = torch.sum(f * (w * v2)[None, :], dim=1)
    T = ((E2 / rho - ux*ux) / 3.0).clamp_min(1e-8)

    cx = vx[None, :] - ux[:, None]
    cy = vy[None, :]
    cz = vz[None, :]
    c2 = cx*cx + cy*cy + cz*cz

    qx = torch.sum(f * w[None, :] * (0.5*c2*cx), dim=1)
    sig = torch.sum(f * w[None, :] * (cx*cx - c2/3.0), dim=1)

    return rho, ux, T, qx, sig


def maxwellian_target(vx, vy, vz, rho, ux, T):
    cx = vx[None, :] - ux[:, None]
    cy = vy[None, :]
    cz = vz[None, :]
    c2 = cx*cx + cy*cy + cz*cz
    pref = rho[:, None] / ((2.0*np.pi*T[:, None])**1.5)
    M = pref * torch.exp(-0.5*c2/T[:, None])
    return M, cx, c2


def shakhov_target(vx, vy, vz, rho, ux, T, qx, Pr, clip_floor=None):
    M, cx, c2 = maxwellian_target(vx, vy, vz, rho, ux, T)

    # Standard monatomic Shakhov correction in R=1 units, q_x = 1/2 int c^2 c_x f dv.
    # fS = M [1 + (1-Pr) (c_x q_x)/(5 rho T^2) (c^2/T - 5)]
    denom = 5.0 * rho[:, None] * T[:, None]**2
    phi = (1.0 - Pr) * (cx * qx[:, None]) / (denom + 1e-30) * (c2/T[:, None] - 5.0)
    fS = M * (1.0 + phi)

    if clip_floor is not None:
        fS = torch.clamp(fS, min=clip_floor)

    return fS


def boundary_maxwellian(vx, vy, vz, rho, ux, T):
    rho_t = torch.tensor([rho], device=vx.device, dtype=vx.dtype)
    ux_t = torch.tensor([ux], device=vx.device, dtype=vx.dtype)
    T_t = torch.tensor([T], device=vx.device, dtype=vx.dtype)
    M, _, _ = maxwellian_target(vx, vy, vz, rho_t, ux_t, T_t)
    return M[0]


def compute_high_moments_chunked(f, vx, vy, vz, w, rho, ux, T, chunk=64):
    device = f.device
    nx, nv = f.shape

    R = torch.zeros(nx, device=device)
    Delta = torch.zeros(nx, device=device)

    for a in range(0, nx, chunk):
        b = min(nx, a+chunk)
        fb = f[a:b]
        rb = rho[a:b]
        ub = ux[a:b]
        Tb = T[a:b]

        M, cx, c2 = maxwellian_target(vx, vy, vz, rb, ub, Tb)
        df = fb - M

        kr = c2 * (cx*cx - c2/3.0)
        c4 = c2*c2

        sig = torch.sum(fb * w[None, :] * (cx*cx - c2/3.0), dim=1)
        R[a:b] = torch.sum(df * w[None, :] * kr, dim=1) - 7.0*Tb*sig
        Delta[a:b] = torch.sum(df * w[None, :] * c4, dim=1)

    return R, Delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="BGK fullstate npz")
    ap.add_argument("--out", required=True, help="output Shakhov fullstate npz")
    ap.add_argument("--Pr", type=float, default=2.0/3.0)
    ap.add_argument("--Kn", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=1200)
    ap.add_argument("--omega", type=float, default=0.65)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip-target", action="store_true")
    ap.add_argument("--floor", type=float, default=1e-30)
    ap.add_argument("--moment-chunk", type=int, default=64)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU")
        device = "cpu"
    else:
        device = args.device

    z = np.load(args.init, allow_pickle=True)

    x_np = np.asarray(z[key(z, ["x", "x_mfp"])]).astype(np.float32)
    nx = len(x_np)
    vx_np, vy_np, vz_np, w_np = load_velocity_grid(z)
    nv = len(w_np)

    f_np = orient_f(z[key(z, ["f", "F", "dist", "distribution"])], nx, nv).astype(np.float32)

    x = torch.tensor(x_np, device=device)
    vx = torch.tensor(vx_np, device=device)
    vy = torch.tensor(vy_np, device=device)
    vz = torch.tensor(vz_np, device=device)
    w = torch.tensor(w_np, device=device)

    f = torch.tensor(f_np, device=device)
    f = torch.clamp(f, min=args.floor)

    dx = float(np.mean(np.diff(x_np)))
    invKn = 1.0 / args.Kn

    # Boundary states from the initial plateaux.
    rho0, ux0, T0, q0, sig0 = torch_moments(f, vx, vy, vz, w)
    nedge = max(10, int(0.05*nx))
    rhoL = float(torch.mean(rho0[:nedge]).detach().cpu())
    uxL  = float(torch.mean(ux0[:nedge]).detach().cpu())
    TL   = float(torch.mean(T0[:nedge]).detach().cpu())
    rhoR = float(torch.mean(rho0[-nedge:]).detach().cpu())
    uxR  = float(torch.mean(ux0[-nedge:]).detach().cpu())
    TR   = float(torch.mean(T0[-nedge:]).detach().cpu())

    fL = boundary_maxwellian(vx, vy, vz, rhoL, uxL, TL)
    fR = boundary_maxwellian(vx, vy, vz, rhoR, uxR, TR)

    pos = vx > 1e-12
    neg = vx < -1e-12
    zer = ~(pos | neg)

    vpos = vx[pos]
    vneg = vx[neg]

    coef_pos = vpos/dx
    denom_pos = coef_pos + invKn

    coef_neg = (-vneg)/dx
    denom_neg = coef_neg + invKn

    print("="*80)
    print("[Shakhov source iteration]")
    print("init:", args.init)
    print("out :", args.out)
    print("device:", device)
    print("nx,nv:", nx, nv)
    print("dx:", dx, "Kn:", args.Kn, "Pr:", args.Pr)
    print("left state :", rhoL, uxL, TL)
    print("right state:", rhoR, uxR, TR)
    print("max_iter:", args.max_iter, "omega:", args.omega)
    print("clip target:", args.clip_target)
    print("="*80, flush=True)

    last_rho = rho0.detach().clone()

    for it in range(1, args.max_iter+1):
        rho, ux, T, qx, sig = torch_moments(f, vx, vy, vz, w)

        fS = shakhov_target(
            vx, vy, vz, rho, ux, T, qx,
            Pr=args.Pr,
            clip_floor=args.floor if args.clip_target else None
        )

        fnew = torch.empty_like(f)

        # Positive velocities: sweep left to right.
        if pos.any():
            prev = fL[pos]
            for i in range(nx):
                val = (coef_pos*prev + invKn*fS[i, pos]) / denom_pos
                fnew[i, pos] = val
                prev = val

        # Negative velocities: sweep right to left.
        if neg.any():
            prev = fR[neg]
            for i in range(nx-1, -1, -1):
                val = (coef_neg*prev + invKn*fS[i, neg]) / denom_neg
                fnew[i, neg] = val
                prev = val

        # Zero velocities.
        if zer.any():
            fnew[:, zer] = fS[:, zer]

        if args.clip_target:
            fnew = torch.clamp(fnew, min=args.floor)

        f = (1.0-args.omega)*f + args.omega*fnew

        if it == 1 or it % args.log_every == 0 or it == args.max_iter:
            rho2, ux2, T2, qx2, sig2 = torch_moments(f, vx, vy, vz, w)
            drho = torch.max(torch.abs(rho2-last_rho)).item()
            last_rho = rho2.detach().clone()

            # crude center from density midpoint
            rmid = 0.5*(float(torch.mean(rho2[:nedge])) + float(torch.mean(rho2[-nedge:])))
            idx = int(torch.argmin(torch.abs(rho2-rmid)).detach().cpu())
            print(
                f"[iter {it:06d}] max|drho|={drho:.3e} "
                f"center={float(x[idx]):+.5f} "
                f"rhoL={float(torch.mean(rho2[:nedge])):.6g} "
                f"rhoR={float(torch.mean(rho2[-nedge:])):.6g} "
                f"q_peak={float(torch.max(torch.abs(qx2))):.6g} "
                f"Tmin={float(torch.min(T2)):.6g}",
                flush=True
            )

    rho, ux, T, qx, sig = torch_moments(f, vx, vy, vz, w)
    Rxx, Delta = compute_high_moments_chunked(
        f, vx, vy, vz, w, rho, ux, T, chunk=args.moment_chunk
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    v_np = np.stack([vx_np, vy_np, vz_np], axis=1)

    np.savez_compressed(
        out,
        x=x_np,
        rho=rho.detach().cpu().numpy(),
        ux=ux.detach().cpu().numpy(),
        T=T.detach().cpu().numpy(),
        qx=qx.detach().cpu().numpy(),
        sigma_xx=sig.detach().cpu().numpy(),
        Rxx_closure=Rxx.detach().cpu().numpy(),
        Delta=Delta.detach().cpu().numpy(),
        S_R_plus_Delta3=(Rxx + Delta/3.0).detach().cpu().numpy(),
        f=f.detach().cpu().numpy().astype(np.float32),
        v=v_np.astype(np.float32),
        w=w_np.astype(np.float32),
        Pr=np.array(args.Pr),
        Kn=np.array(args.Kn),
        left_state=np.array([rhoL, uxL, TL], dtype=np.float64),
        right_state=np.array([rhoR, uxR, TR], dtype=np.float64),
    )

    print("[done] wrote", out)


if __name__ == "__main__":
    main()
