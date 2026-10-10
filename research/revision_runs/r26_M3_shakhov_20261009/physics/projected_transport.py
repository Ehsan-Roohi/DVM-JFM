"""Generated exact 1D projection of Eq29 f5; see derive_projection.py.
Kinetic transport only, with exact conserved-flux gradients eliminated.
"""
import numpy as np
from numpy import *
def transport(rho,u,T,sig,q,m,R,D,phi,psi,om,du,dT,dm,dR,dD,dphi,dpsi,dom):
    return np.stack([
        -1/5*(-9*T*du*rho + 9*dT*rho*u - 5*dm + 5*du*rho*u**2 - 9*du*sig),
        (1/30)*(30*T**2*du*rho + 45*T*dT*rho*u - 60*T*du*rho*u**2 + 5*dD*u + 15*dR*u - 45*dT*rho*u**3 + 105*dT*sig*u + 30*du*m*u + 96*du*q*u)/u,
        (1/175)*(315*T**2*du*rho - 315*T*dT*rho*u - 315*T*du*rho*u**2 + 45*dR*u + 315*dT*sig*u + 175*dm*u**2 + 175*dphi*u + 490*du*m*u + 168*du*q*u + 315*du*sig*u**2)/u,
        (1/315)*(196*D*du + 1065*R*du - 1176*T**2*du*rho - 1764*T*dT*rho*u + 630*T*dm + 1134*T*du*sig + 315*dR*u + 2835*dT*m + 2352*dT*q + 2205*dT*sig*u + 84*dom + 315*dpsi + 630*du*m*u + 630*du*phi + 1176*du*q*u),
        (1/3)*(7*D*du + 12*R*du - 24*T**2*du*rho - 36*T*dT*rho*u + 3*dD*u + 84*dT*q + 3*dom + 24*du*q*u)
    ],axis=-1)
def constitutive(rho,u,T,sig,q,m,R,D,du,dT,dm,dR,dD):
    return np.stack([
        -16/245*(9*R*du + 35*T*dm + 63*T*du*sig + 35*dT*m + 35*du*m*u)/rho,
        -1/175*(810*R*dT + 405*R*du*u + 405*T*dR + 2835*T*dT*sig + 1610*T*du*m + 1512*T*du*q + 1575*dT*m*u)/rho,
        -1/15*(105*D*dT + 35*D*du*u + 270*R*dT + 60*R*du*u + 35*T*dD + 60*T*dR + 420*T*dT*sig + 120*T*du*m + 504*T*du*q + 420*dT*q*u)/rho
    ],axis=-1)
