#!/usr/bin/env python
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normal_shock_states(M1=2.0, gamma=5.0/3.0, rho1=1.0, T1=1.0):
    a1 = math.sqrt(gamma*T1)
    u1 = M1*a1

    r = ((gamma+1.0)*M1*M1)/((gamma-1.0)*M1*M1 + 2.0)
    p2_p1 = 1.0 + 2.0*gamma/(gamma+1.0)*(M1*M1 - 1.0)

    rho2 = rho1*r
    T2 = T1*p2_p1/r
    u2 = u1/r

    return {
        "rho1": rho1, "u1": u1, "T1": T1,
        "rho2": rho2, "u2": u2, "T2": T2,
        "gamma": gamma, "M1": M1
    }


def make_vgrid(nvx, nvy, nvz, vmax, device, dtype):
    vx = torch.linspace(-vmax, vmax, nvx, device=device, dtype=dtype)
    vy = torch.linspace(-vmax, vmax, nvy, device=device, dtype=dtype)
    vz = torch.linspace(-vmax, vmax, nvz, device=device, dtype=dtype)

    def trap_w(n, h):
        w = torch.ones(n, device=device, dtype=dtype)*h
        w[0] *= 0.5
        w[-1] *= 0.5
        return w

    wx = trap_w(nvx, 2*vmax/(nvx-1))
    wy = trap_w(nvy, 2*vmax/(nvy-1))
    wz = trap_w(nvz, 2*vmax/(nvz-1))

    VX, VY, VZ = torch.meshgrid(vx, vy, vz, indexing="ij")
    W = wx[:, None, None]*wy[None, :, None]*wz[None, None, :]

    v = torch.stack([VX.reshape(-1), VY.reshape(-1), VZ.reshape(-1)], dim=1)
    w = W.reshape(-1)
    v2 = v[:,0]**2 + v[:,1]**2 + v[:,2]**2
    return v, v2, w


def moments(f, v, v2, w):
    rho = torch.sum(f*w[None, :], dim=1)
    momx = torch.sum(f*v[None, :, 0]*w[None, :], dim=1)
    ux = momx/(rho + 1.0e-30)

    e2 = torch.sum(f*v2[None, :]*w[None, :], dim=1)
    T = (e2/(rho + 1.0e-30) - ux**2)/3.0
    T = torch.clamp(T, min=1.0e-10)

    cx = v[None, :, 0] - ux[:, None]
    cy = v[None, :, 1]
    cz = v[None, :, 2]
    c2 = cx**2 + cy**2 + cz**2

    qx = 0.5*torch.sum(c2*cx*f*w[None, :], dim=1)
    sig = torch.sum((cx**2 - c2/3.0)*f*w[None, :], dim=1)

    return rho, ux, T, qx, sig


def conservative_discrete_maxwellian(v, v2, w, rho_target, u_target, T_target, niter=5):
    """
    Construct a discrete Maxwellian on the given velocity grid whose quadrature
    moments match rho_target, u_target, and T_target.

    Unknown parameters are u_p and T_p in the Maxwellian exponent. Density is
    then scaled linearly to match rho. Newton iterations solve the discrete mean
    and temperature constraints.
    """
    device = v.device
    dtype = v.dtype
    rho_target = rho_target.to(device=device, dtype=dtype)
    u_target = u_target.to(device=device, dtype=dtype)
    T_target = torch.clamp(T_target.to(device=device, dtype=dtype), min=1.0e-8)

    u = u_target.clone()
    logT = torch.log(T_target.clone())

    vx = v[:, 0]
    two_pi = torch.tensor(2.0*math.pi, device=device, dtype=dtype)

    for _ in range(niter):
        T = torch.exp(logT)
        cx = vx[None, :] - u[:, None]
        c2 = cx**2 + v[None, :, 1]**2 + v[None, :, 2]**2

        logM0 = -1.5*torch.log(two_pi*T[:, None]) - c2/(2.0*T[:, None])
        M0 = torch.exp(logM0)

        Z0 = torch.sum(M0*w[None, :], dim=1)
        Z1 = torch.sum(M0*vx[None, :]*w[None, :], dim=1)
        Z2 = torch.sum(M0*v2[None, :]*w[None, :], dim=1)

        mean = Z1/(Z0 + 1.0e-30)
        temp = (Z2/(Z0 + 1.0e-30) - mean**2)/3.0

        r1 = mean - u_target
        r2 = temp - T_target

        dlog_du = cx/T[:, None]
        dlog_dL = -1.5 + c2/(2.0*T[:, None])  # L = log(T)

        dZ0u = torch.sum(M0*dlog_du*w[None, :], dim=1)
        dZ1u = torch.sum(M0*vx[None, :]*dlog_du*w[None, :], dim=1)
        dZ2u = torch.sum(M0*v2[None, :]*dlog_du*w[None, :], dim=1)

        dZ0L = torch.sum(M0*dlog_dL*w[None, :], dim=1)
        dZ1L = torch.sum(M0*vx[None, :]*dlog_dL*w[None, :], dim=1)
        dZ2L = torch.sum(M0*v2[None, :]*dlog_dL*w[None, :], dim=1)

        invZ0 = 1.0/(Z0 + 1.0e-30)

        dmean_du = dZ1u*invZ0 - mean*dZ0u*invZ0
        dmean_L  = dZ1L*invZ0 - mean*dZ0L*invZ0

        dE_du = dZ2u*invZ0 - (Z2*invZ0)*dZ0u*invZ0
        dE_L  = dZ2L*invZ0 - (Z2*invZ0)*dZ0L*invZ0

        dtemp_du = (dE_du - 2.0*mean*dmean_du)/3.0
        dtemp_L  = (dE_L  - 2.0*mean*dmean_L)/3.0

        # Solve J delta = -r, where variables are [u, logT].
        a = dmean_du
        b = dmean_L
        c = dtemp_du
        d = dtemp_L
        det = a*d - b*c
        det = torch.where(torch.abs(det) < 1.0e-14, torch.sign(det + 1.0e-30)*1.0e-14, det)

        du = (-r1*d + b*r2)/det
        dL = (c*r1 - a*r2)/det

        du = torch.clamp(du, -0.25, 0.25)
        dL = torch.clamp(dL, -0.25, 0.25)

        u = u + du
        logT = logT + dL

    T = torch.exp(logT)
    cx = vx[None, :] - u[:, None]
    c2 = cx**2 + v[None, :, 1]**2 + v[None, :, 2]**2

    logM0 = -1.5*torch.log(two_pi*T[:, None]) - c2/(2.0*T[:, None])
    M0 = torch.exp(logM0)
    Z0 = torch.sum(M0*w[None, :], dim=1)

    scale = rho_target/(Z0 + 1.0e-30)
    M = scale[:, None]*M0
    return torch.clamp(M, min=1.0e-30)


def state_maxwellian(v, v2, w, rho, u, T):
    rho_t = torch.tensor([rho], device=v.device, dtype=v.dtype)
    u_t = torch.tensor([u], device=v.device, dtype=v.dtype)
    T_t = torch.tensor([T], device=v.device, dtype=v.dtype)
    return conservative_discrete_maxwellian(v, v2, w, rho_t, u_t, T_t, niter=8)[0]


def find_crossing(x, rho, rho_mid):
    xr = x.detach().cpu().numpy()
    rr = rho.detach().cpu().numpy()
    y = rr - rho_mid
    idx = np.where(y[:-1]*y[1:] <= 0)[0]
    if len(idx) == 0:
        return 0.0
    k = idx[np.argmin(np.abs(xr[idx]))]
    x0, x1 = xr[k], xr[k+1]
    y0, y1 = y[k], y[k+1]
    if abs(y1-y0) < 1.0e-14:
        return float(x0)
    return float(x0 - y0*(x1-x0)/(y1-y0))


def recenter_f(f, x, shift, ML, MR):
    nx, nv = f.shape
    dx = x[1] - x[0]
    xp = x + shift
    s = (xp - x[0])/dx
    i0 = torch.floor(s).long()
    a = (s - i0.to(s.dtype)).clamp(0, 1)

    out = torch.empty_like(f)
    left = i0 < 0
    right = i0 >= nx-1
    mid = ~(left | right)

    if torch.any(mid):
        ii = i0[mid]
        aa = a[mid][:, None]
        out[mid] = (1-aa)*f[ii] + aa*f[ii+1]
    if torch.any(left):
        out[left] = ML[None, :]
    if torch.any(right):
        out[right] = MR[None, :]
    return out


def save_npz(path, x_scaled, x_mfp, rho, ux, T, qx, sig, states, xhalf_mfp):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=x_scaled,
        x_mfp=x_mfp,
        rho=rho,
        ux=ux,
        T=T,
        qx=qx,
        sigma_xx=sig,
        sig=sig,
        states=np.array([
            states["rho1"], states["u1"], states["T1"],
            states["rho2"], states["u2"], states["T2"]
        ]),
        xhalf_mfp=np.array(xhalf_mfp),
        kn_eff=np.array(1.0/(2.0*xhalf_mfp)),
    )
    print(f"[saved] {path}")


def plot_npz(path, out, title):
    z = np.load(path)
    x = z["x"]
    keys = [
        ("rho", r"$\rho$"),
        ("ux", r"$u_x$"),
        ("T", r"$T$"),
        ("qx", r"$q_x$"),
        ("sigma_xx", r"$\sigma_{xx}$"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(9.0, 10.5), sharex=True)
    for ax, (k, lab) in zip(axes, keys):
        ax.plot(x, z[k], lw=2.2)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"scaled coordinate $x/(2L_\lambda)$")
    fig.suptitle(title)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"[plot] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref/standing_M2_conservative_dvm.npz")
    ap.add_argument("--fig", default="figures/standing_M2_conservative_dvm_profiles.png")
    ap.add_argument("--M1", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=5.0/3.0)
    ap.add_argument("--rho1", type=float, default=1.0)
    ap.add_argument("--T1", type=float, default=1.0)
    ap.add_argument("--xhalf-mfp", type=float, default=30.0)
    ap.add_argument("--nx", type=int, default=1200)
    ap.add_argument("--nvx", type=int, default=48)
    ap.add_argument("--nvy", type=int, default=18)
    ap.add_argument("--nvz", type=int, default=18)
    ap.add_argument("--vmax", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--cfl", type=float, default=0.35)
    ap.add_argument("--center-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--corr-iters", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"[device] {device}, dtype={dtype}")

    states = normal_shock_states(args.M1, args.gamma, args.rho1, args.T1)
    print("[states]", states)

    v, v2, w = make_vgrid(args.nvx, args.nvy, args.nvz, args.vmax, device, dtype)
    print(f"[vgrid] Nv={v.shape[0]} = {args.nvx}x{args.nvy}x{args.nvz}, vmax={args.vmax}")

    ML = state_maxwellian(v, v2, w, states["rho1"], states["u1"], states["T1"])
    MR = state_maxwellian(v, v2, w, states["rho2"], states["u2"], states["T2"])

    for name, M in [("left", ML), ("right", MR)]:
        rho, ux, T, qx, sig = moments(M[None, :], v, v2, w)
        print(
            f"[quad check] {name}: rho={float(rho):.8f}, ux={float(ux):.8f}, T={float(T):.8f}, "
            f"q={float(qx):.3e}, sig={float(sig):.3e}"
        )

    x = torch.linspace(-args.xhalf_mfp, args.xhalf_mfp, args.nx, device=device, dtype=dtype)
    dx = float((x[1]-x[0]).detach().cpu())
    dt = args.cfl*dx/args.vmax
    print(f"[x] [-{args.xhalf_mfp},{args.xhalf_mfp}], nx={args.nx}, dx={dx:.4e}, dt={dt:.4e}")

    H = 0.5*(1.0 + torch.tanh(x/3.0))
    rho0 = (1-H)*states["rho1"] + H*states["rho2"]
    ux0 = (1-H)*states["u1"] + H*states["u2"]
    T0 = (1-H)*states["T1"] + H*states["T2"]

    f = conservative_discrete_maxwellian(v, v2, w, rho0, ux0, T0, niter=args.corr_iters)

    pos = v[:, 0] > 0
    neg = v[:, 0] < 0
    rho_mid = 0.5*(states["rho1"] + states["rho2"])
    last_rho = None

    running_out = str(Path(args.out).with_name(Path(args.out).stem + "_running.npz"))

    for step in range(1, args.steps+1):
        f_left = torch.cat([ML[None, :], f[:-1]], dim=0)
        f_right = torch.cat([f[1:], MR[None, :]], dim=0)

        df_pos = (f - f_left)/dx
        df_neg = (f_right - f)/dx
        df = torch.where((v[:,0] >= 0)[None, :], df_pos, df_neg)

        fstar = f - dt*v[None, :, 0]*df
        fstar[0, pos] = ML[pos]
        fstar[-1, neg] = MR[neg]
        fstar = torch.clamp(fstar, min=1.0e-30)

        rho, ux, T, qx, sig = moments(fstar, v, v2, w)
        Mloc = conservative_discrete_maxwellian(v, v2, w, rho, ux, T, niter=args.corr_iters)

        f = Mloc + (fstar - Mloc)*math.exp(-dt)
        f = torch.clamp(f, min=1.0e-30)
        f[0, pos] = ML[pos]
        f[-1, neg] = MR[neg]

        if step % args.center_every == 0:
            rho, ux, T, qx, sig = moments(f, v, v2, w)
            xs = find_crossing(x, rho, rho_mid)
            if abs(xs) > 1.0e-10:
                f = recenter_f(f, x, xs, ML, MR)

        if step == 1 or step % args.save_every == 0 or step == args.steps:
            rho, ux, T, qx, sig = moments(f, v, v2, w)
            xs = find_crossing(x, rho, rho_mid)

            maxdrho = 0.0
            if last_rho is not None:
                maxdrho = float(torch.max(torch.abs(rho-last_rho)).detach().cpu())
            last_rho = rho.detach().clone()

            print(
                f"[step] {step:7d}/{args.steps} center={xs:+.4e} "
                f"rhoL={float(rho[0]):.6f} rhoR={float(rho[-1]):.6f} "
                f"uL={float(ux[0]):.6f} uR={float(ux[-1]):.6f} "
                f"TL={float(T[0]):.6f} TR={float(T[-1]):.6f} "
                f"maxdrho={maxdrho:.3e}",
                flush=True
            )

            x_mfp = x.detach().cpu().numpy()
            x_scaled = x_mfp/(2.0*args.xhalf_mfp)
            save_npz(
                running_out,
                x_scaled,
                x_mfp,
                rho.detach().cpu().numpy(),
                ux.detach().cpu().numpy(),
                T.detach().cpu().numpy(),
                qx.detach().cpu().numpy(),
                sig.detach().cpu().numpy(),
                states,
                args.xhalf_mfp
            )

    rho, ux, T, qx, sig = moments(f, v, v2, w)
    xs = find_crossing(x, rho, rho_mid)
    f = recenter_f(f, x, xs, ML, MR)
    rho, ux, T, qx, sig = moments(f, v, v2, w)

    x_mfp = x.detach().cpu().numpy()
    x_scaled = x_mfp/(2.0*args.xhalf_mfp)

    save_npz(
        args.out,
        x_scaled,
        x_mfp,
        rho.detach().cpu().numpy(),
        ux.detach().cpu().numpy(),
        T.detach().cpu().numpy(),
        qx.detach().cpu().numpy(),
        sig.detach().cpu().numpy(),
        states,
        args.xhalf_mfp
    )

    title = f"Conservative DVM BGK normal shock, M1={args.M1}, xhalf={args.xhalf_mfp} mfp"
    plot_npz(args.out, args.fig, title)


if __name__ == "__main__":
    main()
