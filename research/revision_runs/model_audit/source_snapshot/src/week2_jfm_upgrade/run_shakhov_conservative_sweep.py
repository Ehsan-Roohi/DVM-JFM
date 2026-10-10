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


def load_grid(z):
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


def moments(f, vx, vy, vz, v2, w):
    rho = torch.sum(f * w[None, :], dim=1).clamp_min(1e-30)
    momx = torch.sum(f * (w * vx)[None, :], dim=1)
    ux = momx / rho

    e2 = torch.sum(f * (w * v2)[None, :], dim=1)
    T = ((e2 / rho - ux * ux) / 3.0).clamp_min(1e-9)

    cx = vx[None, :] - ux[:, None]
    cy = vy[None, :]
    cz = vz[None, :]
    c2 = cx*cx + cy*cy + cz*cz

    psi_q = 0.5 * c2 * cx
    qx = torch.sum(f * w[None, :] * psi_q, dim=1)
    sig = torch.sum(f * w[None, :] * (cx*cx - c2/3.0), dim=1)

    return rho, ux, T, qx, sig


def discrete_maxwellian_chunk(vx, v2, w, rho, ux, T, maxit=7):
    """
    M = exp(a0 + a1 vx + a2 |v|^2)
    matches rho, rho ux, rho(ux^2+3T) on the discrete grid.
    """
    pi = torch.tensor(np.pi, dtype=vx.dtype, device=vx.device)

    a2 = -0.5 / T
    a1 = ux / T
    a0 = torch.log(rho / ((2.0*pi*T)**1.5)) - 0.5 * ux*ux / T

    target0 = rho
    target1 = rho * ux
    target2 = rho * (ux*ux + 3.0*T)

    for _ in range(maxit):
        arg = a0[:, None] + a1[:, None] * vx[None, :] + a2[:, None] * v2[None, :]
        arg = torch.clamp(arg, min=-90.0, max=60.0)
        M = torch.exp(arg)

        wM = w[None, :] * M

        m0 = torch.sum(wM, dim=1)
        m1 = torch.sum(wM * vx[None, :], dim=1)
        m2 = torch.sum(wM * v2[None, :], dim=1)

        m11 = torch.sum(wM * (vx*vx)[None, :], dim=1)
        m12 = torch.sum(wM * (vx*v2)[None, :], dim=1)
        m22 = torch.sum(wM * (v2*v2)[None, :], dim=1)

        F = torch.stack([m0-target0, m1-target1, m2-target2], dim=1)

        J = torch.zeros((len(rho), 3, 3), dtype=vx.dtype, device=vx.device)
        J[:, 0, 0] = m0
        J[:, 0, 1] = m1
        J[:, 0, 2] = m2
        J[:, 1, 0] = m1
        J[:, 1, 1] = m11
        J[:, 1, 2] = m12
        J[:, 2, 0] = m2
        J[:, 2, 1] = m12
        J[:, 2, 2] = m22

        delta = torch.linalg.solve(J, F[:, :, None]).squeeze(-1)

        # Mild damping prevents rare Newton overshoots.
        a0 = a0 - delta[:, 0]
        a1 = a1 - delta[:, 1]
        a2 = a2 - delta[:, 2]

    arg = a0[:, None] + a1[:, None] * vx[None, :] + a2[:, None] * v2[None, :]
    arg = torch.clamp(arg, min=-90.0, max=60.0)
    M = torch.exp(arg)
    return M


def conservative_shakhov_target_chunk(vx, vy, vz, v2, w, rho, ux, T, qx, Pr, q_tol=1e-14):
    """
    Discrete-conservative Shakhov target:

    fS = M_d + h

    M_d matches rho, rho u, energy exactly.
    h has zero mass, momentum, energy on the discrete grid.
    h has heat flux qS = (1-Pr) qx exactly on the discrete grid.

    h is built from a Shakhov-like basis and projected to the invariant nullspace.
    """
    M = discrete_maxwellian_chunk(vx, v2, w, rho, ux, T)

    cx = vx[None, :] - ux[:, None]
    cy = vy[None, :]
    cz = vz[None, :]
    c2 = cx*cx + cy*cy + cz*cz
    psi = 0.5 * c2 * cx

    # Raw Shakhov direction without q amplitude.
    b0 = M * cx * (c2 / T[:, None] - 5.0)

    # Project b0 to zero mass, momentum_x, energy.
    # Basis for invariants: 1, vx, |v|^2.
    phi0 = torch.ones_like(vx)
    phi1 = vx
    phi2 = v2

    wM = w[None, :] * M

    # Gram matrix G_ij = <phi_i, M phi_j>.
    G = torch.zeros((len(rho), 3, 3), dtype=vx.dtype, device=vx.device)

    phis = [phi0, phi1, phi2]
    for i in range(3):
        for j in range(3):
            G[:, i, j] = torch.sum(wM * (phis[i]*phis[j])[None, :], dim=1)

    C = torch.zeros((len(rho), 3), dtype=vx.dtype, device=vx.device)
    for i in range(3):
        C[:, i] = torch.sum(w[None, :] * phis[i][None, :] * b0, dim=1)

    lam = torch.linalg.solve(G, C[:, :, None]).squeeze(-1)
    poly = lam[:, 0, None] + lam[:, 1, None]*vx[None, :] + lam[:, 2, None]*v2[None, :]
    b = b0 - M * poly

    hb = torch.sum(w[None, :] * psi * b, dim=1)
    q_target = (1.0 - Pr) * qx

    scale = torch.zeros_like(qx)
    good = torch.abs(hb) > q_tol
    scale[good] = q_target[good] / hb[good]

    h = scale[:, None] * b
    fS = M + h

    # Diagnostics
    qS = torch.sum(w[None, :] * psi * fS, dim=1)
    m0S = torch.sum(w[None, :] * fS, dim=1)
    m1S = torch.sum(w[None, :] * vx[None, :] * fS, dim=1)
    eS = torch.sum(w[None, :] * v2[None, :] * fS, dim=1)

    min_fS = torch.min(fS)
    return fS, qS, m0S, m1S, eS, min_fS


def build_target(f, vx, vy, vz, v2, w, Pr, chunk):
    nx = f.shape[0]
    rho, ux, T, qx, sig = moments(f, vx, vy, vz, v2, w)

    fS = torch.empty_like(f)

    qS_all = torch.zeros(nx, dtype=f.dtype, device=f.device)
    m0err = torch.zeros(nx, dtype=f.dtype, device=f.device)
    m1err = torch.zeros(nx, dtype=f.dtype, device=f.device)
    eerr = torch.zeros(nx, dtype=f.dtype, device=f.device)
    min_target = torch.tensor(1e30, dtype=f.dtype, device=f.device)

    for a in range(0, nx, chunk):
        b = min(nx, a+chunk)
        out = conservative_shakhov_target_chunk(
            vx, vy, vz, v2, w,
            rho[a:b], ux[a:b], T[a:b], qx[a:b],
            Pr=Pr
        )
        fSb, qSb, m0S, m1S, eS, min_fS = out
        fS[a:b] = fSb
        qS_all[a:b] = qSb
        m0err[a:b] = m0S - rho[a:b]
        m1err[a:b] = m1S - rho[a:b]*ux[a:b]
        eerr[a:b] = eS - rho[a:b]*(ux[a:b]*ux[a:b] + 3*T[a:b])
        min_target = torch.minimum(min_target, min_fS)

    return fS, rho, ux, T, qx, sig, qS_all, m0err, m1err, eerr, min_target


def boundary_discrete_maxwellian(vx, vy, vz, v2, w, rho, ux, T):
    r = torch.tensor([rho], dtype=vx.dtype, device=vx.device)
    u = torch.tensor([ux], dtype=vx.dtype, device=vx.device)
    tt = torch.tensor([T], dtype=vx.dtype, device=vx.device)
    return discrete_maxwellian_chunk(vx, v2, w, r, u, tt)[0]


def high_moments_chunked(f, vx, vy, vz, v2, w, rho, ux, T, sig, chunk):
    nx = f.shape[0]
    R = torch.zeros(nx, dtype=f.dtype, device=f.device)
    D = torch.zeros(nx, dtype=f.dtype, device=f.device)

    for a in range(0, nx, chunk):
        b = min(nx, a+chunk)
        M = discrete_maxwellian_chunk(vx, v2, w, rho[a:b], ux[a:b], T[a:b])
        cx = vx[None, :] - ux[a:b, None]
        cy = vy[None, :]
        cz = vz[None, :]
        c2 = cx*cx + cy*cy + cz*cz
        df = f[a:b] - M
        kr = c2*(cx*cx - c2/3.0)
        R[a:b] = torch.sum(w[None, :] * kr * df, dim=1) - 7.0*T[a:b]*sig[a:b]
        D[a:b] = torch.sum(w[None, :] * c2*c2 * df, dim=1)

    return R, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Pr", type=float, default=2.0/3.0)
    ap.add_argument("--Kn", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-chunk", type=int, default=32)
    ap.add_argument("--moment-chunk", type=int, default=32)
    ap.add_argument("--floor", type=float, default=1e-30)
    ap.add_argument("--allow-negative", action="store_true", help="Do not clamp f to positivity; useful for Shakhov tails.")
    ap.add_argument("--use-rh-boundaries", action="store_true",
                    help="Use exact monatomic Rankine-Hugoniot states instead of plateau states.")
    ap.add_argument("--M1", type=float, default=3.0,
                    help="Upstream Mach number for RH boundaries.")
    ap.add_argument("--gamma", type=float, default=5.0/3.0,
                    help="Specific heat ratio for RH boundaries.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU")
        device = "cpu"
    else:
        device = args.device

    z = np.load(args.init, allow_pickle=True)

    x_np = np.asarray(z[key(z, ["x", "x_mfp"])]).astype(np.float32)
    nx = len(x_np)

    vx_np, vy_np, vz_np, w_np = load_grid(z)
    nv = len(w_np)

    f_np = orient_f(z[key(z, ["f", "F", "dist", "distribution"])], nx, nv).astype(np.float32)

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    vx = torch.tensor(vx_np, dtype=torch.float32, device=device)
    vy = torch.tensor(vy_np, dtype=torch.float32, device=device)
    vz = torch.tensor(vz_np, dtype=torch.float32, device=device)
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    v2 = vx*vx + vy*vy + vz*vz

    f = torch.tensor(f_np, dtype=torch.float32, device=device)
    if not args.allow_negative:
        f = torch.clamp(f, min=args.floor)

    dx = float(np.mean(np.diff(x_np)))
    invKn = 1.0 / args.Kn

    rho0, ux0, T0, q0, sig0 = moments(f, vx, vy, vz, v2, w)

    nedge = max(10, int(0.05*nx))
    rhoL = float(torch.mean(rho0[:nedge]).detach().cpu())
    uxL  = float(torch.mean(ux0[:nedge]).detach().cpu())
    TL   = float(torch.mean(T0[:nedge]).detach().cpu())

    rhoR = float(torch.mean(rho0[-nedge:]).detach().cpu())
    uxR  = float(torch.mean(ux0[-nedge:]).detach().cpu())
    TR   = float(torch.mean(T0[-nedge:]).detach().cpu())

    if args.use_rh_boundaries:
        g = args.gamma
        M1 = args.M1
        rhoL = 1.0
        TL = 1.0
        uxL = M1 * float(np.sqrt(g * TL))

        rho_ratio = ((g + 1.0) * M1 * M1) / ((g - 1.0) * M1 * M1 + 2.0)
        p_ratio = 1.0 + (2.0 * g / (g + 1.0)) * (M1 * M1 - 1.0)

        rhoR = rho_ratio
        TR = p_ratio / rho_ratio
        uxR = uxL / rho_ratio

        print("[override] using exact RH boundary states")
        print("[RH left ]", rhoL, uxL, TL)
        print("[RH right]", rhoR, uxR, TR)

    fL = boundary_discrete_maxwellian(vx, vy, vz, v2, w, rhoL, uxL, TL)
    fR = boundary_discrete_maxwellian(vx, vy, vz, v2, w, rhoR, uxR, TR)

    pos = vx > 1e-12
    neg = vx < -1e-12
    zer = ~(pos | neg)

    vpos = vx[pos]
    vneg = vx[neg]

    coef_pos = vpos / dx
    denom_pos = coef_pos + invKn

    coef_neg = (-vneg) / dx
    denom_neg = coef_neg + invKn

    print("="*88)
    print("[Conservative discrete Shakhov sweep]")
    print("init:", args.init)
    print("out :", args.out)
    print("device:", device)
    print("nx,nv:", nx, nv)
    print("dx:", dx, "Kn:", args.Kn, "Pr:", args.Pr)
    print("states L:", rhoL, uxL, TL)
    print("states R:", rhoR, uxR, TR)
    print("max_iter:", args.max_iter, "omega:", args.omega)
    print("target_chunk:", args.target_chunk)
    print("="*88, flush=True)

    last_rho = rho0.detach().clone()

    for it in range(1, args.max_iter+1):
        fS, rho, ux, T, qx, sig, qS, m0err, m1err, eerr, min_target = build_target(
            f, vx, vy, vz, v2, w, args.Pr, args.target_chunk
        )

        fnew = torch.empty_like(f)

        if pos.any():
            prev = fL[pos]
            for i in range(nx):
                val = (coef_pos*prev + invKn*fS[i, pos]) / denom_pos
                fnew[i, pos] = val
                prev = val

        if neg.any():
            prev = fR[neg]
            for i in range(nx-1, -1, -1):
                val = (coef_neg*prev + invKn*fS[i, neg]) / denom_neg
                fnew[i, neg] = val
                prev = val

        if zer.any():
            fnew[:, zer] = fS[:, zer]

        f = (1.0-args.omega)*f + args.omega*fnew

        # Shakhov targets can have small negative high-velocity tails.
        # Clamping f can break high-order moment budgets, so allow disabling it.
        if not args.allow_negative:
            f = torch.clamp(f, min=args.floor)

        if it == 1 or it % args.log_every == 0 or it == args.max_iter:
            rho2, ux2, T2, q2, sig2 = moments(f, vx, vy, vz, v2, w)
            drho = torch.max(torch.abs(rho2-last_rho)).item()
            last_rho = rho2.detach().clone()

            qratio = torch.mean(qS[torch.abs(qx)>1e-10] / (qx[torch.abs(qx)>1e-10] + 1e-30)).item()
            qtarget = 1.0 - args.Pr

            print(
                f"[iter {it:06d}] max|drho|={drho:.3e} "
                f"rhoL={float(torch.mean(rho2[:nedge])):.6g} "
                f"rhoR={float(torch.mean(rho2[-nedge:])):.6g} "
                f"q_peak={float(torch.max(torch.abs(q2))):.6g} "
                f"Tmin={float(torch.min(T2)):.6g} "
                f"qS/q={qratio:.6g} target={qtarget:.6g} "
                f"max_cons=({float(torch.max(torch.abs(m0err))):.1e},"
                f"{float(torch.max(torch.abs(m1err))):.1e},"
                f"{float(torch.max(torch.abs(eerr))):.1e}) "
                f"min_fS={float(min_target):.3e}",
                flush=True
            )

    rho, ux, T, qx, sig = moments(f, vx, vy, vz, v2, w)
    R, D = high_moments_chunked(
        f, vx, vy, vz, v2, w,
        rho, ux, T, sig,
        chunk=args.moment_chunk
    )
    S = R + D/3.0

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
        Rxx_closure=R.detach().cpu().numpy(),
        Delta=D.detach().cpu().numpy(),
        S_R_plus_Delta3=S.detach().cpu().numpy(),
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
