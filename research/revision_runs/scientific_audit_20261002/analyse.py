from pathlib import Path
import numpy as np,json,hashlib,zipfile,datetime,struct
R=Path(__file__).resolve().parent
P=Path('/project/pi_roohie_umass_edu/BGK_shock/revision_20260929_gpu_preempt')
D=Path('/scratch4/workspace/roohie_umass_edu-jfm1439-dvm-resume-20261001/active')
PAIR=Path('/scratch4/workspace/roohie_umass_edu-jfm1439-pairfix-20261001')
U=Path('/scratch4/workspace/roohie_umass_edu-jfm1439-ppc1920-3840-20261001/outputs/dsmc')
SH=Path('/scratch4/workspace/roohie_umass_edu-jfm1439-shakhov-validation-20261001')
sources={}
def readcsv(p):
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    z=np.genfromtxt(p,delimiter=',',names=True)
    assert all(np.isfinite(z[k]).all() for k in z.dtype.names)
    return z
def save(n,o):
    (R/(n+'.json')).write_text(json.dumps(o,indent=2,allow_nan=False));print(n,'complete',flush=True)
def center(x,n,mid=2):
    j=np.where((n[:-1]<mid)&(n[1:]>=mid))[0];assert len(j)==1
    i=j[0];return x[i]+(mid-n[i])*(x[i+1]-x[i])/(n[i+1]-n[i])
def align(x,n,fields):
    c=center(x,n);g=x[abs(x)<=15];assert min(g+c)>=min(x) and max(g+c)<=max(x)
    return {k:np.interp(g+c,x,v) for k,v in fields.items()}
def rel(a,b):return float(100*np.linalg.norm(a-b)/np.linalg.norm(b))
keys=['Rxx_closure','Delta','S','QDelta_collision']
def pairs():
    result={}
    for ppc in [120,960]:
        modes={}
        for mode in ['original','corrected']:
            rows=[]
            for seed in [1,2]:
                d=PAIR/'outputs'/f'{mode}_ppc{ppc}_seed{seed}'
                done=json.loads((d/'DONE.json').read_text());p=d/'M3_final.csv'
                assert hashlib.sha256(p.read_bytes()).hexdigest()==done['sha256']
                z=readcsv(p);rows.append(align(z['x'],z['n'],{k:z[k] for k in keys}))
            modes[mode]={k:np.array([r[k] for r in rows]) for k in keys}
        values={}
        for k in keys:
            a=modes['original'][k];b=modes['corrected'][k]
            delta=b-a;values[k]={'mean_difference_L2_percent':rel(b.mean(0),a.mean(0)),
                'paired_seed_difference_percent':[rel(b[i],a[i]) for i in range(2)],
                'original_between_seed_L2_percent':rel(a[0],a[1]),'corrected_between_seed_L2_percent':rel(b[0],b[1]),
                'paired_effect_direction_cosine':float(np.dot(delta[0],delta[1])/(np.linalg.norm(delta[0])*np.linalg.norm(delta[1])))}
        result[str(ppc)]=values
    save('paired_sampler',{'method':'Same seed labels; linear density-midpoint alignment, |x-xs|<=15; two seeds only. RNG trajectories diverge after algorithm change; paired labels do not guarantee variance reduction. No significance or convergence claim.', 'results':result})
def statistics():
    results={};allhalves={k:[[],[]] for k in keys}; allblocks={k:[] for k in keys}
    for seed in range(1,9):
        snap=[readcsv(U/f'ppc3840_seed{seed}'/f'M3_step{i:08d}.csv') for i in range(80000,180001,10000)]
        blocks=[]
        for a,b in zip(snap[:-1],snap[1:]):
            ns=b['nsamples']-a['nsamples'];assert np.all(ns==1000)
            fields={k:(b[k]*b['nsamples']-a[k]*a['nsamples'])/ns for k in ['n',*keys[:-1]]}
            fields['QDelta_collision']=(b['QDelta_collision']*b['sample_time']-a['QDelta_collision']*a['sample_time'])/(b['sample_time']-a['sample_time'])
            blocks.append(align(a['x'],fields['n'],fields))
        vals={}
        for k in keys:
            v=np.array([b[k] for b in blocks]); halves=[v[:5].mean(0),v[5:].mean(0)]
            for i in [0,1]:allhalves[k][i].append(halves[i])
            allblocks[k].append(v)
            norms=np.linalg.norm(v,axis=1)
            vals[k]={'halves_L2_percent':rel(*halves),'lag1_norm_descriptive':float(np.corrcoef(norms[:-1],norms[1:])[0,1])}
        results[str(seed)]=vals
    aggregate={};rng=np.random.default_rng(20261002)
    for k in keys:
        a,b=map(np.array,allhalves[k]);idx=rng.integers(0,8,(2000,8))
        boots=np.array([rel(a[i].mean(0),b[i].mean(0)) for i in idx])
        v=np.array(allblocks[k]); widths={}
        for blocksize in [1,2,5]:
            grouped=v.reshape(8,10//blocksize,blocksize,-1).mean(2)
            # Diagnostic variance estimate only, conditional on independent groups.
            sem=grouped.std(1,ddof=1)/np.sqrt(10//blocksize)
            widths[str(blocksize*10000)]={'mean_within_seed_SE_norm':float(np.linalg.norm(sem,axis=1).mean()),'groups_per_seed':10//blocksize}
        aggregate[k]={'early_late_ensemble_L2_percent':rel(a.mean(0),b.mean(0)),
            'paired_seed_bootstrap95_percentile':np.quantile(boots,[.025,.975]).tolist(),'block_size_diagnostics':widths}
    save('time_statistics',{'method':'Ten disjoint 10000-step blocks over80000:180000. Moment sample-sum differences; collision-source exposure-time weighted differences. Each block independently shock aligned. Eight seeds are bootstrap units, paired early/late. Percentile interval of nonnegative L2 norm is descriptive, not a test of zero drift. Only ten blocks: no defensible full ESS or independence certification.','seeds':results,'aggregate':aggregate})
def clipping():
    p=SH/'clip_events.jsonl';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()];assert len(rows)>=20
    ks=[k for k in rows[0] if '_over_' in k]
    save('clipping_budget',{'events':len(rows),'maxima':{k:max(q[k] for q in rows) for k in ks},'sums_of_rowwise_maxima_upper_bounds':{k:sum(q[k] for q in rows) for k in ks},'max_clipped_nodes':max(q['clipped_nodes'] for q in rows),'most_negative_preclip':min(q['min_before'] for q in rows),'scope':'Clipping-only moment increments in pre-clip local frame, per event. Sums of spatial maxima are upper bounds, not net production at a fixed cell or residual of converged PDE. Matched20sweep branches remain non-steady diagnostic.'})
def grad(y,axes,order):
    g=list(np.gradient(y,*axes,edge_order=2))
    if order==4:
        for a,x in enumerate(axes):
            h=x[1]-x[0];dest=[slice(None)]*3;dest[a]=slice(2,-2)
            def sl(s):q=[slice(None)]*3;q[a]=s;return y[tuple(q)]
            g[a][tuple(dest)]=(sl(slice(None,-4))-8*sl(slice(1,-3))+8*sl(slice(3,-1))-sl(slice(4,None)))/(12*h)
    return g
def entropy(f,axes,w,beta=-1.25,order=2):
    f=np.asarray(f,dtype=np.float64);w=np.asarray(w,dtype=np.float64);axes=[np.asarray(a,dtype=np.float64) for a in axes];grids=np.meshgrid(*axes,indexing='ij');r=float(np.sum(f*w));u=np.array([np.sum(f*w*g)/r for g in grids]);c=[g-ui for g,ui in zip(grids,u)]
    theta=np.array([[np.sum(f*w*ci*cj)/r for cj in c] for ci in c]);T=np.trace(theta)/3;D=(1-beta)*T*np.eye(3)+beta*theta
    log=np.log(np.maximum(f,1e-300));gg=grad(log,axes,order)
    flux=[f*(c[i]+sum(D[i,j]*gg[j] for j in range(3))) for i in range(3)]
    bulk=float(sum(np.sum(flux[i]*gg[i]*w) for i in range(3)))
    div=sum(grad(flux[i],axes,order)[i] for i in range(3))
    direct=float(-np.sum(div*log*w));boundary=0.
    for i,ax in enumerate(axes):
        for side,sign in [(0,-1),(-1,1)]:
            idx=[slice(None)]*3;idx[i]=side;idx=tuple(idx)
            wf=w[idx]/((ax[1]-ax[0])*.5)
            boundary+=sign*float(np.sum(log[idx]*flux[i][idx]*wf))
    energy=float(np.sum(div*sum(ci*ci for ci in c)*w)/(r*T))
    return {'rho':r,'T':float(T),'min_diffusion_eigenvalue_over_T':float(np.linalg.eigvalsh(D).min()/T),'entropy_direct_over_rho':direct/r,'entropy_by_parts_over_rho':(bulk-boundary)/r,'boundary_entropy_over_rho':boundary/r,'integration_by_parts_defect_over_rho':(direct-bulk+boundary)/r,'mass_source_over_rho':float(np.sum(div*w)/r),'energy_source_over_rhoT':energy}
def distribution_audit():
    results={}
    for name in ['M5_baseline','M5_velocity_extent','M5_velocity_refine']:
        d=(P if name=='M5_baseline' else D)/'outputs/dvm'/name
        if not (d/'DONE.json').exists():results[name]={'status':'unfinished; no final fullstate audit'};continue
        profile=np.load(d/'result.npz');fp=d/'result_fullstate.npz';z=np.load(fp,allow_pickle=False);v=z['v'].astype(np.float64);w=z['w'].astype(np.float64);x=z['x_mfp'].astype(np.float64);axes=[np.unique(v[:,i]) for i in range(3)];shape=tuple(map(len,axes));assert np.prod(shape)==len(w)
        assert np.allclose(np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3),v)
        shift=center(x,profile['rho'],.5*(profile['rho'][0]+profile['rho'][-1]))
        targets=np.r_[x[0],x[-1],shift+np.linspace(-12,12,17),x[np.argmax(abs(profile['Delta_dvm']))]]
        ids=np.unique([np.argmin(abs(x-a)) for a in targets]);rows=[]
        with zipfile.ZipFile(fp) as archive, open(fp,'rb') as h:
            info=archive.getinfo('f.npy');assert info.compress_type==zipfile.ZIP_STORED
            h.seek(info.header_offset);head=struct.unpack('<IHHHHHIIIHH',h.read(30));assert head[0]==0x04034b50
            h.seek(head[-2]+head[-1],1);version=np.lib.format.read_magic(h)
            dims,fortran,dtype=(np.lib.format.read_array_header_1_0(h) if version==(1,0) else np.lib.format.read_array_header_2_0(h));assert not fortran and dims==(len(x),len(w));offset=h.tell();array=np.memmap(fp,mode='r',dtype=dtype,offset=offset,shape=dims)
            for row in ids:
                f=np.asarray(array[int(row)],dtype=np.float64);assert np.isfinite(f).all() and (f>=0).all()
                rho=np.sum(f*w);u=np.sum(f*w*v[:,0])/rho;c=v.copy();c[:,0]-=u;c2=np.sum(c*c,axis=1);T=np.sum(f*w*c2)/(3*rho);weighted=f*w*c2*c2
                # Absolute c^4 contribution on outer boxes; diagnostic fractions, not missing-tail bounds.
                tails={str(frac):float(weighted[np.any(abs(v)>=np.array([max(abs(a)) for a in axes])*frac,axis=1)].sum()/weighted.sum()) for frac in [.7,.8,.9]}
                ff=f.reshape(shape);ww=w.reshape(shape);e2=entropy(ff,axes,ww,order=2);e4=entropy(ff,axes,ww,order=4)
                continuumM=rho/(2*np.pi*T)**1.5*np.exp(-c2/(2*T))
                calibration=entropy(continuumM.reshape(shape),axes,ww,order=4)
                rows.append({'index':int(row),'x_shifted':float(x[row]-shift),'tail_c4_fractions':tails,'raw_Delta':float(weighted.sum()-15*rho*T*T),'discrete_Delta':float(profile['Delta_dvm'][row]),'floor_fraction':float((f<=1.00001e-30).mean()),'entropy_order2':e2,'entropy_order4':e4,'same_grid_continuum_Maxwellian_calibration':calibration})
                print(name,int(row),'entropy',e4['entropy_direct_over_rho'],flush=True)
        results[name]={'status':'selected-row audit','rows':rows,'fullstate_path':str(fp),'f_member_crc32':info.CRC,'f_member_bytes':info.file_size,'profile_sha256':hashlib.sha256((d/'result.npz').read_bytes()).hexdigest()}
        save('distribution_entropy_partial',results)
    save('distribution_entropy',{'scope':'Local ES-FP operator evaluated on BGK reference distributions, not an ES-FP shock solve. Collision rate normalized to1, beta=-1.25. Full temperature tensor. Order2/4 derivatives of log f; boundary flux, conservation defects and same-grid Maxwellian calibration reported. Values are quadrature diagnostics, not a general H theorem. Tail-box fractions are resolved contributions, not certified missing mass beyond bounds.','cases':results})
if __name__=='__main__':
    pairs();statistics();clipping();distribution_audit()
    sources[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save('provenance',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':sources})
