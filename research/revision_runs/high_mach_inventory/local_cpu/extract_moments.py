"""Read-only historical DVM postprocessing; all live writes under scratch."""
import os, json, hashlib, shutil, zipfile, sys
from pathlib import Path
import numpy as np
import torch
from dvm_bgk_normal_shock_conservative import conservative_discrete_maxwellian, moments

ROOT = Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def main(i):
    manifest=json.loads((ROOT/'extraction_manifest.json').read_text())
    case=manifest['cases'][i]; dest=ROOT/case['tag'];dest.mkdir(exist_ok=True)
    if (dest/'DONE.json').exists():
        print('Already finished; inspect existing output', flush=True);return
    provenance={'case':case,'script_sha256':sha(__file__),'solver_sha256':sha(ROOT/'dvm_bgk_normal_shock_conservative.py'),'job':os.environ.get('SLURM_JOB_ID')}
    assert provenance['solver_sha256']==manifest['solver_sha256']
    for key in ['profile','distribution']:
        src=Path(case[key]);target=dest/src.name
        print('Staging',src,flush=True)
        source_hash=sha(src)
        if not target.exists() or sha(target)!=source_hash:
            tmp=target.with_suffix('.copying');shutil.copyfile(src,tmp)
            assert sha(tmp)==source_hash;tmp.replace(target)
        provenance[key+'_sha256']=source_hash
    assert provenance['profile_sha256']==case['profile_sha256']
    z=np.load(dest/Path(case['profile']).name,allow_pickle=False)
    full=dest/Path(case['distribution']).name
    with zipfile.ZipFile(full) as archive:
        provenance['members']=archive.namelist()
        assert all(k+'.npy' in archive.namelist() for k in ['f','v','w'])
        for key in ['f','v','w']:
            with archive.open(key+'.npy') as src, open(dest/(key+'.npy'),'wb') as out: shutil.copyfileobj(src,out,8*1024*1024)
    f=np.load(dest/'f.npy',mmap_mode='r',allow_pickle=False)
    va=np.load(dest/'v.npy',allow_pickle=False);wa=np.load(dest/'w.npy',allow_pickle=False)
    assert f.shape==(len(z['x']),len(wa)) and va.shape==(len(wa),3)
    assert np.isfinite(va).all() and np.isfinite(wa).all() and (wa>0).all()
    torch.set_num_threads(2); device='cpu'
    v=torch.tensor(va,dtype=torch.float64,device=device);w=torch.tensor(wa,dtype=torch.float64,device=device);v2=(v*v).sum(1)
    names=['rho','ux','T','qx','sig','Rxx_closure','Delta_dvm','S_dvm','Delta_raw','Delta_iter12','R_iter12','Md_u_error','Md_T_error']
    result={k:[] for k in names}; minf=float('inf')
    with torch.no_grad():
        for a in range(0,len(f),8):
            fa=np.array(f[a:a+8],dtype=np.float64);assert np.isfinite(fa).all();minf=min(minf,float(fa.min()))
            F=torch.tensor(fa,device=device);rho,u,T,q,sig=moments(F,v,v2,w)
            assert bool((rho>0).all() and (T>0).all())
            cx=v[None,:,0]-u[:,None];c2=cx*cx+v[None,:,1]**2+v[None,:,2]**2
            md=conservative_discrete_maxwellian(v,v2,w,rho,u,T,niter=4)
            md12=conservative_discrete_maxwellian(v,v2,w,rho,u,T,niter=12)
            delta=((F-md)*w*c2**2).sum(1)
            r=((F-md)*w*c2*(cx**2-c2/3)).sum(1)-7*T*sig
            d12=((F-md12)*w*c2**2).sum(1)
            r12=((F-md12)*w*c2*(cx**2-c2/3)).sum(1)-7*T*sig
            _,mu,mt,_,_=moments(md,v,v2,w)
            values=[rho,u,T,q,sig,r,delta,r+delta/3,(F*w*c2**2).sum(1)-15*rho*T*T,d12,r12,mu-u,mt-T]
            for k,val in zip(names,values): result[k].extend(val.cpu().numpy())
            if a%64==0: print('rows',a,'/',len(f),flush=True)
    result={k:np.asarray(val) for k,val in result.items()}
    assert all(np.isfinite(val).all() for val in result.values())
    result.update(x=z['x'],x_mfp=z['x_mfp'],states=z['states'])
    audit={'minimum_f':minf,'shape':list(f.shape),'device':device,'dtype':'float64','reference_iterations':4,'comparison_iterations':12,'identity_max_abs':float(np.max(np.abs(result['S_dvm']-result['Rxx_closure']-result['Delta_dvm']/3)))}
    for key,old in [('rho','rho'),('ux','ux'),('T','T'),('qx','qx'),('sig','sig'),('Rxx_closure','Rxx_closure')]:
        audit[key+'_stored_relative_l2']=float(np.linalg.norm(result[key]-z[old])/max(np.linalg.norm(z[old]),1e-30))
    for key,other in [('Delta_dvm','Delta_iter12'),('Rxx_closure','R_iter12')]:
        audit[key+'_iteration_relative_l2']=float(np.linalg.norm(result[key]-result[other])/max(np.linalg.norm(result[other]),1e-30))
    np.savez(dest/'moments.npz',**result)
    np.savetxt(dest/'moments.csv',np.column_stack([result[k] for k in ['x_mfp']+names]),delimiter=',',header=','.join(['x_mfp']+names),comments='')
    (dest/'audit.json').write_text(json.dumps({'provenance':provenance,'audit':audit},indent=2))
    files=['moments.npz','moments.csv','audit.json']; hashes={n:sha(dest/n) for n in files}
    archive=Path(manifest['archive'])/case['tag'];archive.mkdir(parents=True,exist_ok=True)
    for n in files:
        target=archive/n
        if target.exists(): assert sha(target)==hashes[n], 'Archive collision'
        else: shutil.copyfile(dest/n,target)
        assert sha(target)==hashes[n]
    done={'files':hashes,'archive':str(archive),'audit':audit,'scientific_acceptance':False}
    (dest/'DONE.json').write_text(json.dumps(done,indent=2));shutil.copyfile(dest/'DONE.json',archive/'DONE.json')
    print(json.dumps(done),flush=True)
if __name__=='__main__': main(int(sys.argv[1]))
