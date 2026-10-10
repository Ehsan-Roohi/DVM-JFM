import json,torch,numpy as np,hashlib,sys
from pathlib import Path
import solver
p=Path(__file__).resolve().parent
c=json.loads((p/'config.json').read_text());torch.set_num_threads(2)
z=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
assert z['signature']==solver.sha(p/'solver.py')+solver.sha(p/'config.json')
x,v,w,_,_,left,right,eq=solver.setup(c,'cpu');g=z['g'];h=z['h']
rho,u,T,q=solver.macro(g,h,v,w);G,H=eq(rho,u,T);cx=v[None,:]-u[:,None]
A=(1-c['Pr'])*q[:,None]*cx/(5*rho[:,None]*T[:,None]**2)
gs=G*(1+A*(cx*cx/T[:,None]-3));hs=H*(1+A*(cx*cx/T[:,None]-1))
negative_g=(-torch.minimum(g,torch.zeros_like(g))*w).sum(1)/rho
negative_h=(-torch.minimum(h,torch.zeros_like(h))*w).sum(1)/(h*w).sum(1)
def location(f):
    ix,iv=np.unravel_index(int(torch.argmin(f)),f.shape)
    return {'value':float(f[ix,iv]),'x':float(x[ix]),'vx':float(v[iv]),'rho':float(rho[ix]),'T':float(T[ix]),'q':float(q[ix])}
face=[]
for f,l,r in [(g,left[0],right[0]),(h,left[1],right[1])]:
    lf=torch.cat([l[None,:],f]);rf=torch.cat([f,r[None,:]])
    face.append(torch.where(v[None,:]>=0,lf,rf))
fg,fh=face
fluxes={'mass':(fg*w*v).sum(1),'momentum':(fg*w*v*v).sum(1),'energy':(.5*v*w*(v*v*fg+fh)).sum(1)}
r={'step':z['step'],'negative_g':location(g),'negative_h':location(h),'target_g':location(gs),'target_h':location(hs),'max_negative_mass_fraction':float(negative_g.max()),'max_negative_transverse_energy_fraction':float(negative_h.max()),'numerical_face_flux_relative_span':{k:float((f.max()-f.min())/abs(f[0])) for k,f in fluxes.items()},'source_sha':solver.sha(p/'solver.py'),'checkpoint_sha':solver.sha(p/'checkpoint.pt'),'interpretation':'Signed Shakhov target assessed without clipping. Grid/time comparisons still required; negative marginal is incompatible with a nonnegative full distribution.'}
(p/'distribution_diagnostic.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
