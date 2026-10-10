"""Chunked full-distribution budget audit, no kinetic evolution.

Run only after parent resource decision: --mach 2|3|5 --chunk 4.
Extracts only f.npy with bounded streaming and maps read-only; all dense work
is <=chunk*nv. Uses actual stored velocity coordinates and weights, not a
grid reconstructed from a filename. Four-iteration double target is explicitly
reported as a diagnostic reconstruction, not historical float32/CUDA identity.
"""
import argparse,csv,hashlib,json,os,shutil,time,zipfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def evidence_directory():
    for ancestor in (HERE,*HERE.parents):
        for candidate in (ancestor/'research/revision_runs/model_audit/evidence',ancestor/'publication/research/revision_runs/model_audit/evidence'):
            if (candidate/'5.csv').is_file() and (candidate/'6.csv').is_file() and (candidate/'7.csv').is_file():return candidate
    raise FileNotFoundError('Archive research/revision_runs/model_audit/evidence not found')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def native_ram():
    if os.name=='nt':
        import ctypes
        class M(ctypes.Structure):_fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','avail','tp','ap','tv','av','ex')]
        m=M();m.length=ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError('Invalid RAM')
        return m.avail/2**30
    # This script's current approval is local only; remote needs approved gate.
    raise RuntimeError('Local Windows resource gate required')
def guard():
    if native_ram()<1:raise RuntimeError('RAM below1GiB')
def csv_read(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
def plateau(y):
    ne=max(5,int(.08*len(y)));return y-np.mean(np.r_[y[:ne],y[-ne:]])
def smooth(y):return np.convolve(np.pad(y,(2,2),mode='edge'),np.array([1,4,6,4,1])/16,mode='valid')
def integrate(y,x):
    out=np.zeros_like(y);out[1:]=np.cumsum(.5*(y[1:]+y[:-1])*np.diff(x));return out
def relative(a,b,mask):return float(np.linalg.norm((a-b)[mask])/np.linalg.norm(b[mask]))
def correlation(a,b,mask):return float(np.corrcoef(a[mask],b[mask])[0,1])

def md(v,w,rho,u,T):
    X,Y,Z=v.T;v2=(v*v).sum(1);up=u.copy();lt=np.log(np.maximum(T,1e-8))
    for _ in range(4):
        guard();tp=np.exp(lt);cx=X[None,:]-up[:,None];c2=cx*cx+Y[None,:]**2+Z[None,:]**2
        M=np.exp(-1.5*np.log(2*np.pi*tp[:,None])-c2/(2*tp[:,None]));mw=M*w
        z0=mw.sum(1);mu=(mw*X).sum(1)/z0;e=(mw*v2).sum(1)/z0;temp=(e-mu*mu)/3
        du=cx/tp[:,None];dl=-1.5+c2/(2*tp[:,None])
        def der(weight,score):return (mw*weight*score).sum(1)/z0-(mw*weight).sum(1)/z0*(mw*score).sum(1)/z0
        a=der(X,du);b=der(X,dl);c=(der(v2,du)-2*mu*a)/3;d=(der(v2,dl)-2*mu*b)/3
        r1=mu-u;r2=temp-T;det=a*d-b*c;det=np.where(abs(det)<1e-14,np.sign(det+1e-30)*1e-14,det)
        up+=np.clip((-r1*d+b*r2)/det,-.25,.25);lt+=np.clip((c*r1-a*r2)/det,-.25,.25)
    tp=np.exp(lt);c2=(X[None,:]-up[:,None])**2+Y[None,:]**2+Z[None,:]**2
    M=np.exp(-1.5*np.log(2*np.pi*tp[:,None])-c2/(2*tp[:,None]))
    return np.maximum(M*(rho/(M*w).sum(1))[:,None],1e-30)

def reduce(f,v,w,rho,u,T,target=True):
    X,Y,Z=v.T;cx=X[None,:]-u[:,None];c2=cx*cx+Y[None,:]**2+Z[None,:]**2;fw=f*w
    rf=fw.sum(1);uf=(fw*X).sum(1)/rf;Tf=(fw*c2).sum(1)/(3*rf)
    q=(.5*c2*cx*fw).sum(1);sig=((cx*cx-c2/3)*fw).sum(1);m=((cx**3-.6*cx*c2)*fw).sum(1)
    C4=(c2*c2*fw).sum(1);R=(c2*(cx*cx-c2/3)*fw).sum(1)-7*T*sig
    out={'rho':rf,'u':uf,'T_at_centre':Tf,'q':q,'sig':sig,'m':m,'C4':C4,'D':C4-15*rho*T*T,'R':R,
         'F':(.5*X*c2*cx*fw).sum(1),'J':(X*c2*c2*fw).sum(1),'K':(X*c2*cx*fw).sum(1),
         'A':(X*(.5*c2+cx*cx)*fw).sum(1)}
    if target:
        M=md(v,w,rho,u,T);mw=M*w
        out.update(qMd=(.5*c2*cx*mw).sum(1),C4Md=(c2*c2*mw).sum(1),
                   rhoMd=mw.sum(1),uMd=(X*mw).sum(1)/mw.sum(1),TMd=(c2*mw).sum(1)/(3*mw.sum(1)))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mach',type=int,choices=[2,3,5],required=True);parser.add_argument('--chunk',type=int,default=4);a=parser.parse_args()
    if not 1<=a.chunk<=8:raise ValueError('chunk must be1..8')
    if native_ram()<4:raise RuntimeError('Initial RAM below4GiB')
    start=time.monotonic();path=HERE/'fullstates'/f'BGK_M{a.mach}.npz';outdir=HERE/f'BGK_M{a.mach}_regenerated';outdir.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as arc:
        names=arc.namelist();fkey=next(k for k in ['f.npy','F.npy'] if k in names)
        extracted=outdir/'distribution.npy'
        if not extracted.exists():
            with arc.open(fkey) as src,extracted.open('wb') as dst:
                while True:
                    guard();block=src.read(2**20)
                    if not block:break
                    dst.write(block)
    z=np.load(path,allow_pickle=True);v=np.array(z['v'],dtype=float);w=np.array(z['w'],dtype=float).ravel();x=np.array(z['x_mfp'],dtype=float)
    rho,u,T=[np.array(z[k],dtype=float) for k in ['rho','ux','T']];F=np.load(extracted,mmap_mode='r')
    if F.shape==(len(w),len(x)):F=F.T
    if F.shape!=(len(x),len(w)):raise ValueError('Unknown f orientation')
    # No output broadcast ever exceeds a small chunk; conservative estimate.
    estimated_peak=a.chunk*len(w)*8*30
    if estimated_peak>2**30:raise RuntimeError('Chunk estimated allocation >1GiB')
    stored={};actual={}
    for lo in range(0,len(x),a.chunk):
        guard();hi=min(lo+a.chunk,len(x));f=np.array(F[lo:hi],dtype=float)
        r=reduce(f,v,w,rho[lo:hi],u[lo:hi],T[lo:hi])
        # Recompute all macros and central fields consistently from f.
        rf=r['rho'];uf=r['u'];Tf=((f*w)*((v[:,0][None,:]-uf[:,None])**2+v[:,1][None,:]**2+v[:,2][None,:]**2)).sum(1)/(3*rf)
        s=reduce(f,v,w,rf,uf,Tf)
        for result,dest in [(r,stored),(s,actual)]:
            for key,val in result.items():dest.setdefault(key,np.zeros(len(x)))[lo:hi]=val
        if lo==0 or hi==len(x) or lo%(a.chunk*100)==0:print(json.dumps({'Mach':a.mach,'completed':hi,'total':len(x),'elapsed':time.monotonic()-start}),flush=True)
    old=csv_read(evidence_directory()/f'{ {2:5,3:6,5:7}[a.mach]}.csv')
    np.testing.assert_allclose(x,old['x'],rtol=0,atol=1e-10)
    comparisons={}
    for oldkey,newkey in [('qx','q'),('sigma_xx','sig'),('mxxx','m'),('Delta','D'),('Rxx_closure','R'),('Fq_direct','F')]:
        comparisons[oldkey]={'max_abs':float(np.max(abs(old[oldkey]-stored[newkey]))),'relative_L2':relative(stored[newkey],old[oldkey],np.ones(len(x),bool))}
    oldgeom=np.gradient(smooth(u),x,edge_order=2)*(u*(2.5*rho*T+stored['sig'])+stored['m']+2.2*stored['q'])
    edge=max(5,len(x)//30);oldanchor=np.mean(2.5*rho[:edge]*T[:edge]**2)
    oldF=oldanchor+integrate(-stored['q']-oldgeom,x)
    oldS=2*(oldF-u*stored['q'])-5*rho*T*T-7*T*stored['sig']
    comparisons['S_R_plus_Delta3_rec']={'max_abs':float(np.max(abs(oldS-old['S_R_plus_Delta3_rec']))),'relative_L2':relative(oldS,old['S_R_plus_Delta3_rec'],np.ones(len(x),bool))}
    legacyS=plateau(old['S_R_plus_Delta3_rec']);legacyMask=abs(legacyS)>=.02*np.max(abs(legacyS))
    if legacyMask.sum()<10:legacyMask[:]=True
    outcomes={};arrays={'x':x,'original_heat_S_rec':oldS,'fixed_legacy_mask':legacyMask}
    for name,r,centre,rr,tt in [('stored_centre',stored,u,rho,T),('recomputed_macros',actual,actual['u'],actual['rho'],actual['T_at_centre'])]:
        ux=np.gradient(smooth(centre),x,edge_order=2)
        geom=ux*r['A'];Qq=r['qMd']-r['q']
        F_rec=r['F'][0]+integrate(Qq-geom,x)
        S=r['R']+r['D']/3;S_rec=2*(F_rec-centre*r['q'])-5*rr*tt*tt-7*tt*r['sig']
        # Scalar original clean convention: no smoothing, np.gradient default1.
        forcing=np.gradient(r['J'],x)+4*np.gradient(centre,x)*r['K']
        D_rec=r['C4Md']-forcing-15*rr*tt*tt
        scalar_direct=-(r['C4']-r['C4Md'])
        clean_original=plateau(-forcing)
        mS=abs(S)>.05*np.max(abs(S));display=plateau(S_rec);mask=abs(display)>=.02*np.max(abs(display))
        if mask.sum()<10:mask[:]=True
        outcomes[name]={'S_raw_active5_relL2':relative(S_rec,S,mS),'S_plateau_active2_on_reconstructedS_relL2':relative(display,plateau(S),mask),
          'R_given_directDelta_plateau_relL2':relative(display-plateau(r['D'])/3,plateau(r['R']),mask),
          'R_two_budget_plateau_relL2':relative(display-plateau(D_rec)/3,plateau(r['R']),mask),
          'Delta_budget_plateau_relL2':relative(plateau(D_rec),plateau(r['D']),mask),
          'qMd_max':float(np.max(abs(r['qMd']))),'DeltaMd_max':float(np.max(abs(r['C4Md']-15*rr*tt*tt))),
          'target_mass_error_max':float(np.max(abs(r['rhoMd']-r['rho']))),'target_mean_error_max':float(np.max(abs(r['uMd']-centre)))}
        outcomes[name]['table_metrics_by_mask']={}
        for label,selection in [('fixed_legacy',legacyMask),('corrected_reconstructed_S',mask)]:
            Rp=plateau(r['R']);Dp=plateau(r['D']);Drp=plateau(D_rec)
            outcomes[name]['table_metrics_by_mask'][label]={
                'active_points':int(selection.sum()),'Delta_budget_relL2':relative(Drp,Dp,selection),
                'R_delta0_relL2':relative(display,Rp,selection),'R_fullDelta_relL2':relative(display-Dp/3,Rp,selection),
                'R_budgetDelta_relL2':relative(display-Drp/3,Rp,selection),
                'Delta_budget_correlation':correlation(Drp,Dp,selection),
                'R_budgetDelta_correlation':correlation(display-Drp/3,Rp,selection)}
        scaleS=float(np.linalg.norm(S));sourceEffect=2*integrate(r['qMd'],x)
        outcomes[name].update(anchor_Fq_actual=float(r['F'][0]),anchor_actual_minus_original=float(r['F'][0]-oldanchor),
            anchor_actual_over_upstream_equilibrium_flux=float(r['F'][0]/(2.5*rr[0]*tt[0]**2)),
            source_integrated_S_max=float(np.max(abs(sourceEffect))),
            source_integrated_S_relative_norm=float(np.linalg.norm(sourceEffect)/scaleS),
            source_integrated_S_plateau_relative_norm=float(np.linalg.norm(plateau(sourceEffect))/np.linalg.norm(plateau(S))))
        for key,val in r.items():arrays[name+'_'+key]=val
        arrays.update({name+'_S_rec':S_rec,name+'_D_rec':D_rec,name+'_D_original_rec':clean_original,name+'_scalar_direct':scalar_direct})
    # Archived scalar reproduction checked separately; cannot equate clean/smooth7.
    scalar=csv_read(HERE/f'M{a.mach}_delta_budget.profiles.csv')
    np.testing.assert_allclose(x,scalar['x'],rtol=0,atol=1e-10)
    scalar_comparison={key:float(np.max(abs(stored[new]-scalar[key]))) for key,new in [('C4','C4'),('J_Delta','J'),('K_geom','K')]}
    scalar_comparison['Delta_rec_from_budget_max']=float(np.max(abs(arrays['stored_centre_D_original_rec']-scalar['Delta_rec_from_budget'])))
    scalar_comparison['Delta_original_reference_max']=float(np.max(abs(plateau(stored['D'])-scalar['Delta'])))
    # Exact table convention reproduced from archived CSVs, independent of correction.
    archivedDrec=np.interp(x,scalar['x'],plateau(scalar['Delta_rec_from_budget']))
    Rp=plateau(old['Rxx_closure']);Dp=plateau(old['Delta'])
    original_table={'active_points':int(legacyMask.sum()),'Delta_budget_relL2':relative(archivedDrec,Dp,legacyMask),
        'R_delta0_relL2':relative(legacyS,Rp,legacyMask),'R_fullDelta_relL2':relative(legacyS-Dp/3,Rp,legacyMask),
        'R_budgetDelta_relL2':relative(legacyS-archivedDrec/3,Rp,legacyMask),
        'Delta_budget_correlation':correlation(archivedDrec,Dp,legacyMask),
        'R_budgetDelta_correlation':correlation(legacyS-archivedDrec/3,Rp,legacyMask)}
    np.savez(outdir/'reductions.npz',**arrays)
    report={'status':'FULLSTATE_CHUNK_REDUCTION_NO_EVOLUTION','Mach':a.mach,'input_sha256':digest(path),'f_npy_sha256':digest(extracted),
      'nx':len(x),'nv':len(w),'chunk':a.chunk,'estimated_chunk_bytes':estimated_peak,'target':'actual stored grid/weights; double four-iteration reconstructed Md',
      'tau':1,'coordinate':'x_mfp','heat_original_CSV_reproduction':comparisons,'scalar_original_CSV_reproduction':scalar_comparison,'original_table_mask_metrics':original_table,'corrected_outcomes':outcomes,
      'elapsed_s':time.monotonic()-start,'limitations':'Finite-time steady differential budgets; no evolution or stationarity certification. Reconstructed double target is not original float32/CUDA history. Masks and moment centre explicitly distinguished.'}
    (outdir/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
