"""Independent exact polynomial projection; no printed transport kernel input."""
from pathlib import Path
import sympy as A
import json
OUT=Path(__file__).resolve().parent
c,y,z=A.symbols('c y z',real=True)
rho,u,T,sig,q,m,R,D,phi,psi,om=A.symbols('rho u T sig q m R D phi psi om',real=True)
var=[rho,u,T,sig,q,m,R,D,phi,psi,om]
der=A.symbols('dr du dT ds dq dm dR dD dphi dpsi dom',real=True)
C2=c*c+y*y+z*z;p=rho*T
P2=c*c-C2/3;P3=c**3-A.Rational(3,5)*c*C2
P4=c**4-A.Rational(6,7)*c*c*C2+A.Rational(3,35)*C2**2
G=1+sig*(c*c-(y*y+z*z)/2)/(2*p*T)+c*q/(p*T)*(C2/(5*T)-1)
G+=m*(c**3-A.Rational(3,2)*c*(y*y+z*z))/(6*p*T*T)
G+=R*(c*c-(y*y+z*z)/2)/(4*p*T*T)*(C2/(7*T)-1)
G+=D/(8*p*T)*(C2*C2/(15*T*T)-2*C2/(3*T)+1)
G+=phi*A.Rational(35,8)*P4/(24*p*T**3)
G+=psi*A.Rational(5,2)*P3/(12*p*T**3)*(C2/(9*T)-1)
G+=c*om/(40*p*T*T)*(C2*C2/(7*T*T)-2*C2/T+5)

def E(expr):
    pp=A.Poly(A.expand(expr),c,y,z);ans=0
    for powers,coef in pp.terms():
        if any(n%2 for n in powers):continue
        factor=A.prod(A.factorial2(n-1)*T**(n//2) for n in powers)
        ans+=coef*factor
    return A.factor(ans)

# Derivative at fixed LAB velocity, c=v_x-u(x); Maxwellian log derivative.
dG=der[0]*A.diff(G,rho)-der[1]*A.diff(G,c)+der[2]*A.diff(G,T)
for v,d in zip(var[3:],der[3:]):dG+=d*A.diff(G,v)
df=(der[0]/rho-A.Rational(3,2)*der[2]/T+c*der[1]/T+C2*der[2]/(2*T*T))*G+dG
weights=[P2,c*C2/2,P3,P2*(C2-7*T),C2*C2-15*T*T,P4,P3*(C2-9*T),c*C2*(C2-14*T)]
rows=[]
for k,w in enumerate(weights):
    row=E(rho*w*(u+c)*df);rows.append(row);print('derived',k,flush=True)
# Flux constraints eliminate rho,sig,q spatial derivatives exactly.
flux={der[0]:-rho*der[1]/u,
      der[3]:-rho*u*der[1]+rho*T*der[1]/u-rho*der[2],
      der[4]:-rho*u*(u*der[1]+A.Rational(5,2)*der[2])-der[1]*sig-u*(-rho*u*der[1]+rho*T*der[1]/u-rho*der[2])}
rows=[A.factor(x.subs(flux,simultaneous=True)) for x in rows]
closure=[A.factor(-x.subs({phi:0,psi:0,om:0,der[8]:0,der[9]:0,der[10]:0})/rho) for x in rows[5:]]
out='''"""Generated exact 1D projection of Eq29 f5; see derive_projection.py.
Kinetic transport only, with exact conserved-flux gradients eliminated.
"""
import numpy as np
from numpy import *
def transport(rho,u,T,sig,q,m,R,D,phi,psi,om,du,dT,dm,dR,dD,dphi,dpsi,dom):
    return np.stack([
'''
from sympy.printing.numpy import NumPyPrinter
printer=NumPyPrinter()
out+=',\n'.join('        '+printer.doprint(x).replace('numpy.','np.') for x in rows[:5])+'\n    ],axis=-1)\n'
out+='def constitutive(rho,u,T,sig,q,m,R,D,du,dT,dm,dR,dD):\n    return np.stack([\n'
out+=',\n'.join('        '+printer.doprint(x).replace('numpy.','np.') for x in closure)+'\n    ],axis=-1)\n'
(OUT/'physics/projected_transport.py').write_text(out,encoding='utf-8')
(OUT/'projection_formulae.json').write_text(json.dumps({'variables':list(map(str,var)),'derivatives':list(map(str,der)),'rows':list(map(str,rows)),'closures':list(map(str,closure))},indent=2),encoding='utf-8')
print('DONE',flush=True)
