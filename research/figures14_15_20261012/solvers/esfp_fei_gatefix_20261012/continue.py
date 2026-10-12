import argparse,ctypes,json,time,sys,signal,os
from pathlib import Path
import numpy as np
import fp,shakhov_reference as ref
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--gate',action='store_true');args=ap.parse_args();mach=5 if args.index==0 else 4
stopped=False
def stop(*a):
    global stopped
    stopped=True
signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
names=['continue.py','fp.py','fp_kernel.c','fp_kernel.so','shakhov_reference.py','resource_watchdog.py']
binding={n:ref.sha(P/n) for n in names}
out=P/f'production_M{mach}';out.mkdir(exist_ok=True);cp=out/'checkpoint.npz';z=np.load(cp if cp.exists() else P/f'refined_seed_M{mach}.npz',allow_pickle=False)
c=json.loads(str(z['config']));f=np.ascontiguousarray(z['f']);step=int(z['step']);x,v,w,_,l,r=fp.setup(c)
if cp.exists():assert str(z['binding'])==json.dumps(binding,sort_keys=True)
dt=c['CFL']*2*c['L']/c['nx']/c['vmax'];physical_time=float(z['time']) if 'time' in z else step*dt
if args.gate:
    # Unchanged serial kernel is the oracle for both successful and failed maps.
    old=P.parent/'shock_tol4_esfp_20261011';oracle=ctypes.CDLL(str(old/'fp_kernel.so'));oracle.collide.argtypes=fp.LIB.collide.argtypes;oracle.collide.restype=ctypes.c_int
    results=[]
    for label,ff in [('refined checkpoint',f),('equilibrium',ref.eq(v,np.ones(8),np.ones(8)*3,np.ones(8)))]:
        ff=np.ascontiguousarray(ff);a=np.empty_like(ff);b=np.empty_like(ff);da=np.zeros(5);db=np.zeros(5)
        t=time.monotonic();ca=oracle.collide(ff.shape[1],len(v),v,w[0],dt,ff,a,da);serial=time.monotonic()-t
        t=time.monotonic();cb=fp.LIB.collide(ff.shape[1],len(v),v,w[0],dt,ff,b,db);parallel=time.monotonic()-t
        assert ca==cb==0 and np.array_equal(a,b) and np.array_equal(da,db)
        results.append(dict(fixture=label,outputs_bitwise=True,diagnostics_bitwise=True,serial_seconds=serial,parallel_seconds=parallel))
    # Deliberately inadmissible energy must stop both implementations.
    bad=f[:,:1,:].copy();bad[1]*=-1e3;aa=np.empty_like(bad);bb=np.empty_like(bad);d=np.zeros(5)
    ca=oracle.collide(1,len(v),v,w[0],dt,bad,aa,d);cb=fp.LIB.collide(1,len(v),v,w[0],dt,bad,bb,d)
    assert ca==cb!=0
    new,d,bal=fp.advance(f,v,w,l,r,c,dt);a=ref.moments(f,v,w);b=ref.moments(new,v,w)
    assert np.min(new)>=0 and np.min(b[:,0])>0 and np.min(b[:,2])>0 and bal<3e-11
    # Trigger the published beta bound on an admissible gas covariance.
    u=np.array([3.]);rho=np.array([1.]);Tx=np.array([2.7]);Ty=np.array([.15]);ga=rho[:,None]/np.sqrt(2*np.pi*Tx[:,None])*np.exp(-(v[None,:]-u[:,None])**2/(2*Tx[:,None]))
    anisotropic=np.ascontiguousarray(np.array([ga,2*Ty[:,None]*ga,8*Ty[:,None]**2*ga]))
    fixture_attempts=[]
    for fixture_nv in [c['nv']*2**j for j in range(4) if c['nv']*2**j<=51200]:
        fixture_dv=2*c['vmax']/fixture_nv
        fixture_v=-c['vmax']+(np.arange(fixture_nv)+.5)*fixture_dv
        fixture_w=np.full(fixture_nv,fixture_dv)
        ga=rho[:,None]/np.sqrt(2*np.pi*Tx[:,None])*np.exp(-(fixture_v[None,:]-u[:,None])**2/(2*Tx[:,None]))
        anisotropic=np.ascontiguousarray(np.array([ga,2*Ty[:,None]*ga,8*Ty[:,None]**2*ga]))
        try:
            m=ref.moments(anisotropic,fixture_v,fixture_w);bb,di=fp.collision(anisotropic,fixture_v,fixture_w,.05);n=ref.moments(bb,fixture_v,fixture_w)
        except FloatingPointError as error:
            fixture_attempts.append(dict(nv=fixture_nv,error=str(error)))
            if 'collision guard code 2000;' not in str(error):raise
            continue
        fixture_attempts.append(dict(nv=fixture_nv,fit=di[2],PASS=True));break
    else:raise FloatingPointError('adaptive fixture fails unchanged fit gate through Nv=51200')
    beta=max(-1.25,-(1-1e-12)*m[0,2]/(m[0,4]/m[0,0]));tau=2*(1-beta)/m[0,0]
    assert beta>-1.25 and di[3]>0 and di[4]<2e-11 and np.min(bb)>=0
    assert abs(n[0,4]-m[0,4]/(1+.05*m[0,0]))<1e-10
    results.append(dict(fixture='adaptive beta branch',attempts=fixture_attempts,beta=beta,effective_Pr=3/(2*(1-beta)),invariants=di[4],minimum_diffusion=di[3],stress_source_pass=True))
    ref.dump(out/'GATE.json',dict(PASS=True,binding=binding,oracle_sha=ref.sha(old/'fp_kernel.so'),results=results,hard_stop_preserved=True,full_step_balance=bal,scope='Exact kernel equality and one physical map; no convergence'))
    sys.exit(0)
gate=json.loads((out/'GATE.json').read_text());assert gate['PASS'] and gate['binding']==binding
start=time.monotonic();worst=0.;refinements=[];prev=None;last_rhs=None;reason='bounded_wall_budget';minimum_diffusion=float('inf');maxfit=0.;maxPr=2/3;active_beta_cells=0
def save():
    ref.save(cp,f=f,step=step,time=physical_time,config=np.array(json.dumps(c)),binding=np.array(json.dumps(binding,sort_keys=True)))
def refine():
    global f,c,x,v,w,l,r,prev
    if c['nv']>=51200:raise FloatingPointError('maximum bounded velocity refinement reached')
    dv=w[0];before=ref.moments(f,v,w);ff=np.repeat(f,2,axis=2);h=ff[1].copy();ff[1]-=dv*dv/16*ff[0]
    if np.any(ff[0][h==0]!=0):raise FloatingPointError('nonrealizable empty transverse bin')
    ratio=np.divide(ff[1],h,out=np.ones_like(h),where=h!=0);ff[2]*=ratio**2
    if not np.isfinite(ff).all() or np.min(ff)<0 or np.min(ratio)<=0:raise FloatingPointError('refinement positivity')
    c=dict(c,nv=2*c['nv']);x,v,w,_,l,r=fp.setup(c);after=ref.moments(ff,v,w)
    error=float(np.max(abs(after[:,:3]-before[:,:3])/np.maximum(1,abs(before[:,:3]))))
    if error>1e-11:raise FloatingPointError('refinement invariant defect '+str(error))
    refinements.append(dict(step=step,time=physical_time,nv=c['nv'],invariant_error=error,eight_field_relative_changes=[float(np.max(abs(after[:,j]-before[:,j]))/max(1e-8,np.max(abs(before[:,j])))) for j in range(8)]))
    ref.dump(out/'REFINEMENTS.json',refinements);f=np.ascontiguousarray(ff);prev=None;save()
try:
    while not stopped and time.monotonic()-start<20500:
        try:new,d,bal=fp.advance(f,v,w,l,r,c,dt)
        except FloatingPointError as e:
            # ONLY the unchanged numerical 2% fit guard may trigger velocity
            # refinement. Diffusion, positivity, conservation and resources stop.
            if 'collision guard code 2000;' not in str(e):raise
            refine();continue
        worst=max(worst,bal);minimum_diffusion=min(minimum_diffusion,d[3]);maxfit=max(maxfit,d[2])
        if (step+1)%200==0:
            a=ref.moments(new,v,w);old=ref.moments(f,v,w);scale=np.maximum(1e-8,np.max(abs(a),axis=0));last_rhs=float(np.max(abs(a-old)/(dt*scale)))
            drift=None if prev is None else float(np.max(abs(a-prev)/scale));prev=a
            anis=np.maximum(a[:,4]/a[:,0],-a[:,4]/(2*a[:,0])); beta=np.full(len(a),-1.25); mask=anis>0; beta[mask]=np.maximum(beta[mask],-(1-1e-12)*a[mask,2]/anis[mask]); maxPr=max(maxPr,float(np.max(3/(2*(1-beta))))); active_beta_cells=max(active_beta_cells,int(np.count_nonzero(beta>-1.25)))
            row=dict(max_effective_Pr=maxPr,active_beta_cells=active_beta_cells,step=step+1,time=physical_time+dt,nv=c['nv'],rhs=last_rhs,drift=drift,maximum_balance=worst,minimum_diffusion=minimum_diffusion,max_relative_fit=maxfit,seconds=time.monotonic()-start)
            ref.dump(out/'PROGRESS.json',row);print(json.dumps(row),flush=True)
            if last_rhs<1e-4:f=new;step+=1;physical_time+=dt;reason='user_rhs_threshold_reached';break
        f=new;step+=1;physical_time+=dt
        if step%200==0:save()
    if stopped:reason='signal_checkpoint'
    save();a=ref.moments(f,v,w)
    try:
        extra,d,bal=fp.advance(f,v,w,l,r,c,dt);last_rhs=float(np.max(abs(ref.moments(extra,v,w)-a)/(dt*np.maximum(1e-8,np.max(abs(a),axis=0)))))
    except FloatingPointError:
        raise
    ref.save(out/'profiles.npz',x=x,fields=a,columns=np.array(['rho','u','T','q','sigma','R','Delta','S']))
    ref.dump(out/'STATUS.json',dict(reason=reason,mach=mach,step=step,time=physical_time,nv=c['nv'],final_rhs=last_rhs,converged_user_rhs=last_rhs<1e-4,maximum_balance=worst,minimum_diffusion=minimum_diffusion,max_relative_fit=maxfit,max_effective_Pr=maxPr,active_beta_cells=active_beta_cells,model='Fei Eq25 adaptive ESFP; tau=2*(1-beta)/rho; mu=T; Pr=2/3 only on fixed-beta branch',checkpoint_sha=ref.sha(cp),binding=binding,scope='User RHS convergence only; velocity/domain/time accuracy remains to qualify'))
except BaseException as e:
    save();ref.dump(out/'FAILED.json',dict(error=repr(e),step=step,time=physical_time,nv=c['nv'],checkpoint_sha=ref.sha(cp)));raise
