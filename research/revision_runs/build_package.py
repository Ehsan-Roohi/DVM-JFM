from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'source'

def replace_once(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)

s = (SRC/'dvm_bgk_normal_shock_conservative_hmom_densemicro.py').read_text()
s = replace_once(s, 'import math\n', 'import math\nimport os\nimport json\n')
s = replace_once(s, '\n\ndef higher_moments(', '''

# Limit row-batched temporary tensors; velocity quadrature and update are unchanged.
_moments_original = moments
_maxwell_original = conservative_discrete_maxwellian
def moments(f, v, v2, w):
    pieces = [_moments_original(f[i:i+128], v, v2, w) for i in range(0, f.shape[0], 128)]
    return tuple(torch.cat([p[j] for p in pieces]) for j in range(5))

def conservative_discrete_maxwellian(v, v2, w, rho, ux, T, niter=5):
    return torch.cat([_maxwell_original(v, v2, w, rho[i:i+128], ux[i:i+128],
                      T[i:i+128], niter=niter) for i in range(0, len(rho), 128)])

def higher_moments(''')
s = replace_once(s, '    args = ap.parse_args()', '''    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--vmax-transverse", type=float, default=None)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    signature = {k: v for k, v in vars(args).items() if k not in
                 ("out", "fig", "checkpoint", "checkpoint_every", "steps", "device", "save_every")}
    Path(args.out + ".settings.json").write_text(json.dumps(vars(args), indent=2))''')
s = replace_once(s, '    print(f"[vgrid] Nv={v.shape[0]}', '''    if args.vmax_transverse is not None:
        axes=[]; weights=[]
        for n,bound in [(args.nvx,args.vmax),(args.nvy,args.vmax_transverse),(args.nvz,args.vmax_transverse)]:
            axes.append(torch.linspace(-bound,bound,n,device=device,dtype=dtype))
            ww=torch.ones(n,device=device,dtype=dtype)*(2*bound/(n-1))
            ww[0]*=.5; ww[-1]*=.5; weights.append(ww)
        a,b,c=torch.meshgrid(*axes,indexing="ij")
        v=torch.stack([a.reshape(-1),b.reshape(-1),c.reshape(-1)],dim=1)
        w=(weights[0][:,None,None]*weights[1][None,:,None]*weights[2][None,None,:]).reshape(-1)
        v2=torch.sum(v*v,dim=1)
        print("[transverse vmax]",args.vmax_transverse,flush=True)
    print(f"[vgrid] Nv={v.shape[0]}''')
s = replace_once(s, '    for step in range(1, args.steps+1):', '''    start_step = 0
    if Path(args.checkpoint).exists():
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if ck["signature"] != signature:
            raise RuntimeError("Checkpoint parameters do not match")
        f = ck["f"].to(dtype=dtype)
        last_rho = ck["last_rho"]
        start_step = ck["step"]
        if start_step > args.steps:
            raise RuntimeError("Checkpoint beyond requested end step")
        print(f"[restart] step={start_step}", flush=True)

    for step in range(start_step+1, args.steps+1):''')
needle = '            save_npz(running_out, x_scaled, x_mfp, rho, ux, T, qx, sig, states, args.xhalf_mfp, extra)'
s = replace_once(s, needle, needle + '''
            hist = Path(args.out).parent / "history"
            hist.mkdir(exist_ok=True)
            save_npz(hist / f"step{step:08d}.npz", x_scaled, x_mfp, rho, ux, T, qx, sig, states, args.xhalf_mfp, extra)

        if step % args.checkpoint_every == 0 or step == args.steps:
            if not torch.isfinite(f).all():
                raise RuntimeError("Non-finite distribution")
            tmp = args.checkpoint + ".tmp"
            torch.save({"f": f, "last_rho": last_rho, "step": step,
                        "signature": signature}, tmp)
            os.replace(tmp, args.checkpoint)
            print(f"[checkpoint] step={step}", flush=True)''')
s = replace_once(s, '        "Qxx_norm", "Qx_norm",', '        "Qxx_norm", "Qx_norm", "Delta_dvm", "S_dvm", "Delta_raw",')
s = replace_once(s, '        out["Qx_norm"][sl] = Qx_norm', '''        out["Qx_norm"][sl] = Qx_norm
        delta = torch.sum(Gi * ww * C2**2, dim=1)
        out["Delta_dvm"][sl] = delta
        out["S_dvm"][sl] = Rxx_closure + delta/3
        out["Delta_raw"][sl] = torch.sum(Fi * ww * C2**2, dim=1) - 15*rhoi*Ti**2''')
(SRC/'dvm_revision.py').write_text(s)

s = (SRC/'dsmc_1d_shock_c4.cpp').read_text()
s = replace_once(s, '#include <algorithm>', '#include <algorithm>\n#include <sstream>\n#include <stdexcept>')
fields = 'n ux uy uz T qx sigxx c4 Delta Rxx S JDelta KDelta prod_c4 prod_coll'.split()
io = '''
template<class T> void put(ostream& s, const T& v) { s.write((const char*)&v, sizeof(v)); }
template<class T> void get(istream& s, T& v) { s.read((char*)&v, sizeof(v)); }
void putvec(ostream& s, const vector<double>& v) {
    size_t n=v.size(); put(s,n); s.write((const char*)v.data(), n*sizeof(double));
}
void getvec(istream& s, vector<double>& v) {
    size_t n=0; get(s,n); if(n>100000000) throw runtime_error("Invalid checkpoint vector");
    v.resize(n); s.read((char*)v.data(), n*sizeof(double));
}
string signature(const Args& a) {
    ostringstream s; s<<setprecision(17)<<a.cells<<' '<<a.sample_start<<' '<<a.sample_every
    <<' '<<a.seed<<' '<<a.M1<<' '<<a.gamma<<' '<<a.L<<' '<<a.dt<<' '<<a.ppc
    <<' '<<a.sigmaT<<' '<<a.gmax_factor; return s.str();
}
void save_ck(const Args& a, int step, const Particles& p, const vector<double>& gmax,
 const vector<double>& rem, const ProfileAccum& acc, double time, double st, bool sampling,
 const mt19937_64& rng) {
    string tmp=a.prefix+".checkpoint.tmp"; ofstream o(tmp,ios::binary);
    o<<"JFM1439_C4_V1\\n"<<signature(a)<<'\\n'<<rng<<'\\n';
    put(o,step); put(o,time); put(o,st); put(o,sampling); put(o,acc.nsamp);
    putvec(o,p.x); putvec(o,p.vx); putvec(o,p.vy); putvec(o,p.vz);
    putvec(o,gmax); putvec(o,rem);
    FIELDS_WRITE
    o.flush(); if(!o) throw runtime_error("Checkpoint write failed"); o.close();
    if(rename(tmp.c_str(),(a.prefix+".checkpoint").c_str())) throw runtime_error("Checkpoint rename failed");
}
int load_ck(const Args& a, Particles& p, vector<double>& gmax, vector<double>& rem,
 ProfileAccum& acc, double& time, double& st, bool& sampling, mt19937_64& rng) {
    ifstream i(a.prefix+".checkpoint",ios::binary); if(!i) return 0;
    string magic,sig; getline(i,magic); getline(i,sig);
    if(magic!="JFM1439_C4_V1" || sig!=signature(a)) throw runtime_error("Checkpoint mismatch");
    i>>rng; i.get(); int step=0;
    get(i,step); get(i,time); get(i,st); get(i,sampling); get(i,acc.nsamp);
    getvec(i,p.x); getvec(i,p.vx); getvec(i,p.vy); getvec(i,p.vz);
    getvec(i,gmax); getvec(i,rem);
    FIELDS_READ
    if(!i || step>a.steps || p.x.size()!=p.vx.size() || gmax.size()!=size_t(a.cells))
        throw runtime_error("Invalid checkpoint");
    cout<<"[restart] step="<<step<<endl; return step;
}
'''.replace('FIELDS_WRITE', '\n    '.join(f'putvec(o,acc.{f});' for f in fields)).replace('FIELDS_READ', '\n    '.join(f'getvec(i,acc.{f});' for f in fields))
s = replace_once(s, 'int main(int argc, char** argv) {', io+'\nint main(int argc, char** argv) {')
s = replace_once(s, '    for (int step=1; step<=a.steps; step++) {', '''    int start_step=load_ck(a,p,gmax,rem,acc,time,sample_time,sampling_started,rng);
    for (int step=start_step+1; step<=a.steps; step++) {''')
s = replace_once(s, '        if (step % a.out_every == 0 || step == a.steps) {', '''        if (step % 1000 == 0 || step == a.steps)
            save_ck(a,step,p,gmax,rem,acc,time,sample_time,sampling_started,rng);
        if (step % a.out_every == 0 || step == a.steps) {''')
(SRC/'dsmc_revision.cpp').write_text(s)

# Controlled extent comparison: append whole cells on both sides of each axis.
base = dict(M1=5., nx=2200, nvx=141, nvy=27, nvz=27, vmax=22., xhalf_mfp=80.,
            steps=52000, cfl=.75, center_every=50)
variants = [
 ('M5_baseline', {}),
 ('M5_velocity_extent', dict(vmax=26.4, vmax_transverse=22*16/13,
                            nvx=169, nvy=33, nvz=33, steps=62400, center_every=60)),
 ('M5_velocity_refine', dict(nvx=211, nvy=40, nvz=40)),
 ('M5_space_refine', dict(nx=3299, steps=78000, center_every=75)),
 ('M5_timestep_half', dict(cfl=.375, steps=104000, center_every=100)),
 ('M5_space_extent', dict(xhalf_mfp=120., nx=3300)),
]
# Spatial extent changes delta-x by only 0.015%; report that explicitly.
cases=[]
for name, changes in variants:
    cfg=dict(base, **changes); cfg['name']=name; cases.append(cfg)
(ROOT/'cases.json').write_text(json.dumps(cases, indent=2))
manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SRC.iterdir()) if p.is_file()}
(ROOT/'source_sha256.json').write_text(json.dumps(manifest, indent=2))
print('Built', len(cases), 'DVM cases and restartable DSMC source')
