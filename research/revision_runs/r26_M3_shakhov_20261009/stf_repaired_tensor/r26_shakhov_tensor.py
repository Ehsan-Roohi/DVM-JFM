"""Shakhov collision + first-order Grad26 regularization, isolated diagnostic.
Derived from target polynomial and Gu Appendix A kinetic transport.
All outside-GMM collision rates are 1/tau. Pr=2/3, mu=theta.
Original frozen kernels remain available for explicit Maxwell comparisons.
"""
from types import SimpleNamespace
import dataclasses
import numpy as np
import shock_pilot as base
from shock_pilot import tensors_from_components, R26Gradients, R26ClosureDerivatives, farfields
from r26_state import StateTensors
from r26_tensor_closures import *
from r26_tensor_closures import _validate_tensors_and_gradients, _leading_shape, _broadcast_mu

def closures_from_tensors(
    tensors: StateTensors,
    gradients: R26Gradients,
    *,
    mu: float | np.ndarray = 1.0,
    equation25_mode: str = "v3-literal",
    coefficient_mode: str = "jfm2009",
) -> R26Closures:
    """Evaluate the audited v3 nonlinear phi, psi, and Omega closures."""

    if equation25_mode != "v3-literal":
        raise ValueError("only the source-faithful 'v3-literal' Eq. (25) mode is implemented")
    assert coefficient_mode == 'shakhov-derived'
    coefficients = SimpleNamespace(C1=1.,C2=0.,Y1=1.,Y2=0.,Y3=0.,mode='shakhov-derived',provenance='Shakhov + STF-corrected Gu kinetic transport')
    _validate_tensors_and_gradients(tensors, gradients)
    lead = _leading_shape(tensors)
    muv = _broadcast_mu(mu, lead)

    rho = np.asarray(tensors.rho, dtype=float)
    theta = np.asarray(tensors.theta, dtype=float)
    pressure = rho * theta
    q = np.asarray(tensors.heat_flux, dtype=float)
    sigma = np.asarray(tensors.sigma, dtype=float)
    rr = np.asarray(tensors.R, dtype=float)
    mm = np.asarray(tensors.m, dtype=float)
    delta_moment = np.asarray(tensors.Delta, dtype=float)

    gu = np.asarray(gradients.velocity, dtype=float)
    gs = np.asarray(gradients.sigma, dtype=float)
    gq = np.asarray(gradients.heat_flux, dtype=float)
    gr = np.asarray(gradients.R, dtype=float)
    gm = np.asarray(gradients.m, dtype=float)
    grho = np.asarray(gradients.rho, dtype=float)
    gtheta = np.asarray(gradients.theta, dtype=float)
    gdelta = np.asarray(gradients.Delta, dtype=float)

    div_sigma = np.einsum("...eae->...a", gs)
    div_q = np.einsum("...aa->...", gq)
    div_u = np.einsum("...aa->...", gu)
    # Primary notation: sigma_ml * d_l u_m.  Our gradient storage is
    # gu[derivative, component], whereas ASTR dvel[component, derivative].
    sigma_grad_u = np.einsum("...ab,...ba->...", sigma, gu)

    # Eq. (23), phi_ijkl.  Each raw product is projected in full 3-D.
    raw0 = np.moveaxis(gm, -4, -1) / rho[..., None, None, None, None]
    raw0 -= np.einsum("...abc,...d->...abcd", mm, grho) / rho[..., None, None, None, None] ** 2
    s0 = stf4_project(raw0)
    # sigma_ij d_l u_k and R_ij d_l u_k.  STF4 makes k<->l transposition
    # numerically equivalent, but keep the primary index orientation explicit.
    s1 = stf4_project(np.einsum("...ab,...dc->...abcd", sigma, gu))
    s2 = stf4_project(np.einsum("...abc,...d->...abcd", mm, div_sigma))
    s3 = stf4_project(np.einsum("...ab,...dc->...abcd", rr, gu))
    s4 = stf4_project(np.einsum("...ab,...cd->...abcd", sigma, sigma))
    phi = (
        -4.0 * muv[..., None, None, None, None] / coefficients.C1 * s0
        - 12.0 * muv[..., None, None, None, None] / (coefficients.C1 * rho[..., None, None, None, None]) * s1
        + 4.0 * muv[..., None, None, None, None]
        / (coefficients.C1 * pressure[..., None, None, None, None] * rho[..., None, None, None, None])
        * s2
        - 12.0 * muv[..., None, None, None, None]
        / (7.0 * coefficients.C1 * pressure[..., None, None, None, None])
        * s3
        - coefficients.C2 / (coefficients.C1 * rho[..., None, None, None, None]) * s4
    )

    # Eq. (25), psi_ijk, in the exact v3 source transcription.
    raw0_psi = np.einsum("...cab->...abc", gr) / rho[..., None, None, None]
    raw0_psi -= np.einsum("...ab,...c->...abc", rr, grho) / rho[..., None, None, None] ** 2
    p0 = stf3_project(raw0_psi)
    p1 = stf3_project(
        np.einsum("...ab,...c->...abc", rr + 7.0 * theta[..., None, None] * sigma, gtheta)
    )
    # q_i d_k u_j; STF3 makes j<->k transposition equivalent, but this is the
    # primary orientation for gu[derivative, component].
    p2 = stf3_project(np.einsum("...a,...cb->...abc", q, gu))
    p3 = stf3_project(np.einsum("...ab,...c->...abc", rr, div_sigma))
    # Primary Eq. (25): m_{mij} \partial_k u_m.  ``gu`` stores the
    # derivative index first, gu[k,m] = \partial_k u_m; the audited Fortran
    # oracle stores these two indices in the opposite order.  Keeping this
    # transpose explicit prevents a rotation-covariant but componentwise
    # wrong contraction from slipping through the tensor tests.
    eq25_m_grad_u = 54.0 / 7.0 * np.einsum("...eab,...ce->...abc", mm, gu)
    # Eq25 bracket extends over FOUR indices ijkm, before divergence.
    eq25_plus8_div_u = 8.0 * np.einsum("...dijkd->...ijk", stf4_project(np.einsum("...ijk,...dl->...dijkl", mm, gu)))
    eq25_minus6_div_u = -6.0 * np.einsum("...abc,...->...abc", mm, div_u)
    eq25_raw = eq25_m_grad_u + eq25_plus8_div_u + eq25_minus6_div_u
    p5 = stf3_project(eq25_raw)
    p6 = stf3_project(np.einsum("...ea,...bce->...abc", sigma, mm))
    p7 = stf3_project(np.einsum("...a,...bc->...abc", q, sigma))
    scalar = div_q + sigma_grad_u
    psi = (
        -27.0 * muv[..., None, None, None] / (7.0 * coefficients.Y1) * p0
        - 27.0 * muv[..., None, None, None]
        / (7.0 * coefficients.Y1 * pressure[..., None, None, None])
        * p1
        - 108.0 * muv[..., None, None, None]
        / (5.0 * coefficients.Y1 * rho[..., None, None, None])
        * p2
        + 27.0 * muv[..., None, None, None]
        / (7.0 * coefficients.Y1 * pressure[..., None, None, None] * rho[..., None, None, None])
        * p3
        + 6.0 * muv[..., None, None, None] * scalar[..., None, None, None]
        / (coefficients.Y1 * pressure[..., None, None, None] * rho[..., None, None, None])
        * mm
        - muv[..., None, None, None] / (coefficients.Y1 * rho[..., None, None, None]) * p5
        - coefficients.Y2 / (coefficients.Y1 * rho[..., None, None, None]) * p6
        - coefficients.Y3 / (coefficients.Y1 * rho[..., None, None, None]) * p7
    )

    # Eq. (26), Omega_i, including the v3 removal of the unsupported pressure
    # gradient addition that was present in stage-3 v2.
    div_R_over_rho = np.einsum("...bab->...a", gr) / rho[..., None]
    div_R_over_rho -= np.einsum("...ab,...b->...a", rr, grho) / rho[..., None] ** 2
    t1 = -4.0 * muv[..., None] * div_R_over_rho
    # q_j(d_j u_i + d_i u_j).  The two terms are deliberately written out;
    # unlike a single transpose they form the symmetric velocity gradient.
    # Both Eq26 velocity gradients carry the two-index STF brackets.
    t2_raw = 2.0 * np.einsum("...b,...ab->...a", q, stf2_project(gu))
    t2 = -56.0 * muv[..., None] / (5.0 * rho[..., None]) * t2_raw
    # m_ijk d_k u_j.  m is symmetric in j,k, so a transposed gu would be
    # numerically equivalent; retain the paper orientation explicitly.
    t3 = -8.0 * muv[..., None] / rho[..., None] * np.einsum("...abc,...cb->...a", mm, gu)
    t4_raw = np.einsum("...ab,...b->...a", 2.0 * theta[..., None, None] * sigma + rr, gtheta)
    t4 = -14.0 * muv[..., None] / pressure[..., None] * t4_raw
    t5 = 56.0 * muv[..., None] * q * scalar[..., None] / (3.0 * pressure[..., None] * rho[..., None])
    t6 = 4.0 * muv[..., None] / (pressure[..., None] * rho[..., None]) * np.einsum(
        "...ab,...b->...a", rr, div_sigma
    )
    t7 = (
        7.0
        * muv[..., None]
        * delta_moment[..., None]
        / (3.0 * pressure[..., None])
        * (div_sigma / rho[..., None] - 2.0 * gtheta)
    )
    t8_raw = 14.0 * np.einsum("...b,...ab->...a", q, sigma)
    t8_raw += 5.0 * np.einsum("...abc,...bc->...a", mm, sigma)
    t8 = -2.0 * t8_raw / (15.0 * rho[..., None])
    omega = (
        -7.0
        * muv[..., None]
        / 3.0
        * (gdelta / rho[..., None] - delta_moment[..., None] * grho / rho[..., None] ** 2)
        + t1
        + t2
        + t3
        + t4
        + t5
        + t6
        + t7
        # Shakhov has no quadratic collision coupling.
    )

    if not np.isfinite(phi).all() or not np.isfinite(psi).all() or not np.isfinite(omega).all():
        raise FloatingPointError("R26 closure evaluation produced NaN or infinity")
    return R26Closures(
        phi=phi,
        psi=psi,
        Omega=omega,
        equation25_mode=equation25_mode,
        provenance=coefficients.provenance,
        coefficient_mode=coefficients.mode,
    )

def steady_r26_bulk_residual(t,g,c,cd,*,mu=1.,acceleration=None):
    """Replace only analytically identified Maxwell collision productions."""
    r=base.steady_r26_bulk_residual(t,g,c,cd,mu=mu,acceleration=acceleration)
    nu=t.rho*t.theta/np.asarray(mu)
    sig2=stf2_project(np.einsum('...ki,...jk->...ij',t.sigma,t.sigma))
    norm2=np.einsum('...ij,...ij->...',t.sigma,t.sigma)
    return dataclasses.replace(r,
        m=r.m-.5*nu[...,None,None,None]*t.m,
        R=r.R-nu[...,None,None]*t.R/6-2*nu[...,None,None]*sig2/(3*t.rho[...,None,None]),
        Delta=r.Delta+nu*t.Delta/3-2*nu*norm2/(3*t.rho),
        provenance='Shakhov collision projection + unchanged kinetic R26 transport')

class Shock(base.Shock):
    def evaluate(self,z):
        v=np.asarray(z).reshape(self.n,5)*self.scale
        dv=np.gradient(v,self.x,axis=0,edge_order=2)
        t=self.fields(z);u=t.velocity[:,0];T=t.theta;rho=t.rho
        du=u*dv[:,0];dT=T*dv[:,1];drho=-rho*dv[:,0]
        ds=-self.mass*du-drho*T-rho*dT
        dq=-self.mass*(u*du+2.5*dT)-du*t.sigma[:,0,0]-u*ds
        dt=tensors_from_components(drho,du,dT,dq,ds,dv[:,2],dv[:,3],dv[:,4])
        def g(a):return np.stack([a,np.zeros_like(a),np.zeros_like(a)],axis=1)
        grad=R26Gradients(**{f.name:g(getattr(dt,f.name)) for f in dataclasses.fields(R26Gradients)})
        cl=closures_from_tensors(t,grad,mu=T,coefficient_mode='shakhov-derived')
        dphi=np.gradient(cl.phi,self.x,axis=0,edge_order=2)
        dpsi=np.gradient(cl.psi,self.x,axis=0,edge_order=2)
        do=np.gradient(cl.Omega,self.x,axis=0,edge_order=2)
        cd=R26ClosureDerivatives(dphi[...,0],dpsi[...,0],g(do))
        res=steady_r26_bulk_residual(t,grad,cl,cd,mu=T)
        eq=np.column_stack([res.stress[:,0,0],res.heat_flux[:,0],res.m[:,0,0,0],res.R[:,0,0],res.Delta])
        return t,res,eq
