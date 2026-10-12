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
gate=json.loads((P/'VALIDATION.json').read_text());assert gate['PASS']
for n,h in gate['binding'].items():assert ref.sha(P/n)==h
assert gate['binding']==binding
start=time.monotonic();worst=0.;refinements=[];prev=None;last_rhs=None;reason='bounded_wall_budget';minimum_diffusion=float('inf');maxfit=0.;maxPr=2/3;active_beta_cells=0
def save():
    ref.save(cp,f=f,step=step,time=physical_time,dt=dt,config=np.array(json.dumps(c)),binding=np.array(json.dumps(binding,sort_keys=True)))
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
    while not stopped and time.monotonic()-start<6000:
        try:new,d,bal=fp.advance(f,v,w,l,r,c,dt)
        except FloatingPointError:raise
        worst=max(worst,bal);minimum_diffusion=min(minimum_diffusion,d[3]);maxfit=max(maxfit,d[2])
        if (step+1)%200==0:
            a=ref.moments(new,v,w);old=ref.moments(f,v,w);scale=np.maximum(1e-8,np.max(abs(a),axis=0));last_rhs=float(np.max(abs(a-old)/(dt*scale)))
            drift=None if prev is None else float(np.max(abs(a-prev)/scale));prev=a
            anis=np.maximum(a[:,4]/a[:,0],-a[:,4]/(2*a[:,0])); beta=np.full(len(a),-1.25); mask=anis>0; beta[mask]=np.maximum(beta[mask],-(1-1e-12)*a[mask,2]/anis[mask]); maxPr=max(maxPr,float(np.max(3/(2*(1-beta))))); active_beta_cells=max(active_beta_cells,int(np.count_nonzero(beta>-1.25)))
            row=dict(dt=dt,max_effective_Pr=maxPr,active_beta_cells=active_beta_cells,step=step+1,time=physical_time+dt,nv=c['nv'],rhs=last_rhs,drift=drift,maximum_balance=worst,minimum_diffusion=minimum_diffusion,max_positive_projection=maxfit,seconds=time.monotonic()-start)
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
    ref.dump(out/'STATUS.json',dict(reason=reason,mach=mach,step=step,time=physical_time,nv=c['nv'],final_rhs=last_rhs,converged_user_rhs=last_rhs<1e-4,maximum_balance=worst,minimum_diffusion=minimum_diffusion,max_positive_projection=maxfit,max_effective_Pr=maxPr,active_beta_cells=active_beta_cells,model='Fei Eq25 adaptive ESFP; tau=2*(1-beta)/rho; mu=T; Pr=2/3 only on fixed-beta branch',checkpoint_sha=ref.sha(cp),binding=binding,scope='User RHS convergence only; velocity/domain/time accuracy remains to qualify'))
except BaseException as e:
    save();ref.dump(out/'FAILED.json',dict(error=repr(e),step=step,time=physical_time,nv=c['nv'],checkpoint_sha=ref.sha(cp)));raise
