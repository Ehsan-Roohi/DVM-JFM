"""Independent bounded target/identity controls and integrated precision sensitivity.
Loads only function definitions using AST; never executes parent script body.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
source=HERE/'check_discrete_budget.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
namespace={'np':np}
exec(compile(functions,str(source),'exec'),namespace)
tm=namespace['target_moments'];integrate=namespace['cumtrap'];plateau=namespace['plateau'];norm=namespace['norm']

def dense_target(rho,ut,Tt,nvx,nvy,vmax,niter):
    # Direct three-dimensional implementation of archived double-precision Newton.
    vx=np.linspace(-vmax,vmax,nvx);vy=np.linspace(-vmax,vmax,nvy)
    wx=np.full(nvx,2*vmax/(nvx-1));wy=np.full(nvy,2*vmax/(nvy-1))
    wx[[0,-1]]*=.5;wy[[0,-1]]*=.5
    X,Y,Z=np.meshgrid(vx,vy,vy,indexing='ij')
    w=(wx[:,None,None]*wy[None,:,None]*wy[None,None,:]).ravel()
    X,Y,Z=[a.ravel()[None,:] for a in (X,Y,Z)];v2=X*X+Y*Y+Z*Z
    up=np.asarray(ut).copy();lt=np.log(Tt)
    for _ in range(niter):
        tp=np.exp(lt);cx=X-up[:,None];c2=cx*cx+Y*Y+Z*Z
        M0=np.exp(-1.5*np.log(2*np.pi*tp[:,None])-c2/(2*tp[:,None]))
        mw=M0*w;z0=mw.sum(1);z1=(mw*X).sum(1);z2=(mw*v2).sum(1)
        mu=z1/z0;temp=(z2/z0-mu*mu)/3
        du=cx/tp[:,None];dl=-1.5+c2/(2*tp[:,None])
        def derivative(weight,score):
            return (mw*weight*score).sum(1)/z0-(mw*weight).sum(1)/z0*(mw*score).sum(1)/z0
        a=derivative(X,du);b=derivative(X,dl)
        c=(derivative(v2,du)-2*mu*a)/3;d=(derivative(v2,dl)-2*mu*b)/3
        r1=mu-ut;r2=temp-Tt;det=a*d-b*c
        det=np.where(abs(det)<1e-14,np.sign(det+1e-30)*1e-14,det)
        up+=np.clip((-r1*d+b*r2)/det,-.25,.25)
        lt+=np.clip((c*r1-a*r2)/det,-.25,.25)
    tp=np.exp(lt);cxp=X-up[:,None];c2p=cxp*cxp+Y*Y+Z*Z
    M0=np.exp(-1.5*np.log(2*np.pi*tp[:,None])-c2p/(2*tp[:,None]))
    M=np.maximum(M0*(rho/(M0*w).sum(1))[:,None],1e-30)
    cx=X-ut[:,None];c2=cx*cx+Y*Y+Z*Z;mw=M*w
    return {'q':(.5*c2*cx*mw).sum(1),'delta':(c2*c2*mw).sum(1)-15*rho*Tt*Tt},(X,Y,Z,w,M)

rho=np.array([1.,1.3,.7]);u=np.array([.5,-.4,.9]);T=np.array([1.,.8,1.2])
comparisons=[]
for niter in (4,12):
    sep=tm(rho,u,T,13,9,3.,niter,np.float64)
    den,arrays=dense_target(rho,u,T,13,9,3.,niter)
    errors={k:float(np.max(abs(sep[k]-den[k]))) for k in ('q','delta')}
    for k in errors:np.testing.assert_allclose(sep[k],den[k],rtol=2e-11,atol=2e-12)
    comparisons.append({'iterations':niter,'errors':errors})

X,Y,Z,w,M=arrays
cx=X-u[:,None];c2=cx*cx+Y*Y+Z*Z
f=M*(1+.07*np.sin(X)+.03*np.cos(2*Y))
fw=f*w;mw=M*w
q=(.5*c2*cx*fw).sum(1);qM=(.5*c2*cx*mw).sum(1)
sig=((cx*cx-c2/3)*fw).sum(1)
Bc=(c2*(cx*cx-c2/3)*fw).sum(1);C4=(c2*c2*fw).sum(1)
R=Bc-7*T*sig;D=C4-15*rho*T*T
F=(X*.5*c2*cx*fw).sum(1)
Fcontinuum=u*q+.5*(5*rho*T*T+7*T*sig+R+D/3)
Rd=(c2*(cx*cx-c2/3)*(fw-mw)).sum(1)-7*T*sig
Dd=(c2*c2*(fw-mw)).sum(1);BxxM=(c2*cx*cx*mw).sum(1)
Fdiscrete=u*q+.5*(BxxM+7*T*sig+Rd+Dd/3)
np.testing.assert_allclose(F,Fcontinuum,atol=2e-13)
np.testing.assert_allclose(F,Fdiscrete,atol=2e-13)
rho_f=fw.sum(1);T_f=(c2*fw).sum(1)/(3*rho_f)
m=((cx**3-.6*cx*c2)*fw).sum(1)
A_direct=(X*(.5*c2+cx*cx)*fw).sum(1)
A_moments=u*(2.5*rho_f*T_f+sig)+m+2.2*q
np.testing.assert_allclose(A_direct,A_moments,atol=2e-13)
assert np.max(abs(A_direct-(u*(2.5*rho*T+sig)+m+2.2*q)))>1e-3
assert np.max(abs(BxxM-5*rho*T*T))>.1 # continuum target replacement must fail
assert np.max(abs(qM))>.01 # f=Md has source zero, old -q formula is nonzero

x=np.linspace(-2,2,41);rhs=.3+.2*x;anchor=2.3
actual=anchor+integrate(rhs,x)
np.testing.assert_allclose(actual[0],anchor)
assert np.max(abs(actual-(anchor-integrate(rhs,x))))>.5 # wrong sign rejected
assert np.max(abs(actual-(anchor+.1+integrate(rhs,x))))>.09 # false anchor rejected
scale=80.;scaled=x/scale
np.testing.assert_allclose(integrate(rhs,x),integrate(scale*rhs,scaled),atol=2e-14)
assert np.max(abs(integrate(rhs,x)-integrate(rhs,scaled)))>.5

inputdir=None
for ancestor in (HERE,*HERE.parents):
    for candidate in (ancestor/'research/revision_runs/model_audit/evidence',ancestor/'publication/research/revision_runs/model_audit/evidence'):
        if (candidate/'5.csv').is_file():inputdir=candidate;break
    if inputdir is not None:break
if inputdir is None:raise FileNotFoundError('Archived evidence CSV directory missing')
sensitivity=[]
for mach,index,nvx,nvy,vmax in [(2,5,97,19,12.),(3,6,121,23,16.),(5,7,141,27,22.)]:
    path=inputdir/f'{index}.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    d={k:np.asarray([float(r[k]) for r in rows]) for k in rows[0]}
    rho,u,T=[d[k] for k in ['rho_file','ux_file','T_file']]
    targets={name:tm(rho,u,T,nvx,nvy,vmax,niter,dtype) for name,niter,dtype in [('double4',4,np.float64),('double12',12,np.float64),('float4',4,np.float32)]}
    base=d['S_R_plus_Delta3_rec']+2*(d['Fq_direct'][0]-d['Fq_rec_from_balance'][0])
    S=d['S_R_plus_Delta3_ref'];mask=abs(plateau(S))>.02*np.max(abs(plateau(S)))
    outputs={name:base+2*integrate(target['q'].astype(np.float64),d['x']) for name,target in targets.items()}
    sensitivity.append({'Mach':mach,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'float4_vs_double4_integrated_S_max':float(np.max(abs(outputs['float4']-outputs['double4']))),
        'float4_vs_double4_integrated_S_plateau_max':float(np.max(abs(plateau(outputs['float4'])-plateau(outputs['double4'])))),
        'double12_vs_double4_integrated_S_max':float(np.max(abs(outputs['double12']-outputs['double4']))),
        'S_plateau_active2_relL2_by_target':{name:norm(plateau(arr),plateau(S),mask) for name,arr in outputs.items()}})

result={'status':'PASS_BOUNDED_INDEPENDENT_CONTROLS','reviewed_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'separable_vs_dense_smallgrid':comparisons,'fixture_controls':['continuum decomposition','discrete decomposition','same-centre geometry integral; mixed field moments rejected','bad Bxx target rejected','f=Md old-source rejected','wrong sign rejected','wrong anchor rejected','coordinate scaling and mixed-scale rejection'],
    'integrated_target_sensitivity':sensitivity,
    'limitations':'Factorized float32 arithmetic is not bitwise Torch/CUDA target reproduction; fullstate lineage remains distinct.'}
(HERE/'INDEPENDENT_DISCRETE_BUDGET_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
