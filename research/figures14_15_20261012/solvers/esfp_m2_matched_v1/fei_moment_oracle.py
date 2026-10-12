"""Read-only supplementary oracle for Fei Eqs.23,24,26d,27d.
No production source changes, physical shock steps or checkpoint changes.
Fei Eq.12 R_xx = our S, not our trace-free Rcl_xx.
"""
import json
import numpy as np
import esfp as e
import resource_watchdog as resource
assert resource.memory()['available_bytes']>=4*resource.GIB
records=[]
for nv in [400,800,1600]:
    c=dict(mach=2,nx=12,L=30,nv=nv,vmax=40)
    x,v,w,f,l,rr=e.setup(c);old=e.ref.moments(f,v,w);dt=.01
    new,diag=e.collision(f,v,w,dt);a=e.ref.moments(new,v,w);r=dt*old[:,0]/4.5
    def m3(state,macro):
        cx=v[None,:]-macro[:,1,None]
        return np.sum(cx**3*state[0]*w,axis=1)-6*macro[:,3]/5
    m=m3(f,old);mn=m3(new,a)
    S_expected=(old[:,7]-10*r*a[:,4]**2/a[:,0])/(1+4*r)
    R_expected=(old[:,5]-5*r*a[:,4]**2/a[:,0])/(1+4*r)
    normerr=lambda actual,expected,scale:float(np.max(abs(actual-expected))/max(np.max(abs(scale)),1e-12))
    records.append(dict(nv=nv,mxxx_BE_error=normerr(mn,m/(1+3*r),m),Fei_Rxx_our_S_BE_error=normerr(a[:,7],S_expected,old[:,7]),our_Rcl_BE_error=normerr(a[:,5],R_expected,old[:,5])))
for key in ['mxxx_BE_error','Fei_Rxx_our_S_BE_error','our_Rcl_BE_error']:
    assert records[-1][key]<records[0][key]/8,key
result=dict(PASS=True,tests=records,Fei_Rxx_mapping='Rxx(Fei Eq12)=S(ours)=Rcl_xx+Delta/3',note='Errors are discrete velocity/time approximation diagnostics, not whole-shock accuracy estimates')
e.ref.dump(e.ROOT/'FEI_MOMENT_ORACLE.json',result);print(json.dumps(result))
