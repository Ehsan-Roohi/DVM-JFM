import pathlib,json,hashlib
import numpy as np,pdfplumber
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).resolve().parent;base=root.parent
p=pdfplumber.open(root/'reference.pdf').pages[10]
refs={}
for key,idx,left in [('rho',37,110.257),('T',78,352.777)]:
 a=np.array(p.curves[idx]['pts']);x=-20+(a[:,0]-left)*35/153;y=(227.448-a[:,1])/(227.448-89.662)
 refs[key]=(x,y);np.savetxt(root/f'reference_{key}.csv',np.c_[x,y],delimiter=',',header='x_lambda1,normalized_value',comments='')
# Density-midpoint shift, shared by density and temperature, with no fitted stretch.
def midpoint(x,y):
 i=np.flatnonzero((y[:-1]<=.5)&(y[1:]>=.5))[0];return float(np.interp(.5,y[i:i+2],x[i:i+2]))
xrefmid=midpoint(*refs['rho'])
c=json.loads((base/'config.json').read_text());g=c['gamma'];m=c['M'];r=(g+1)*m*m/((g-1)*m*m+2);T2=(1+2*g/(g+1)*(m*m-1))/r
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
metrics={};styles=[('baseline',base,'#2864b4'),('time_extend',base/'sensitivity/time_extend','#d27916'),('space_refine',base/'sensitivity/space_refine','#299455')]
for name,path,color in styles:
 z=np.load(path/'final.npz');x=z['x'];f=z['fields'];vals={'rho':(f[:,0]-1)/(r-1),'T':(f[:,2]-1)/(T2-1)}
 shift=xrefmid-midpoint(x,vals['rho']);xa=x+shift;metrics[name]={'shift_lambda1':shift,'fields':{}}
 for j,key in enumerate(['rho','T']):
  xr,yr=refs[key];ours=np.interp(xr,xa,vals[key]);err=ours-yr
  metrics[name]['fields'][key]={'max_abs_percent_of_jump':float(100*max(abs(err))),'rms_percent_of_jump':float(100*np.sqrt(np.mean(err**2)))}
  axes[0,j].plot(xa,vals[key],color=color,label=name.replace('_',' '),lw=1.6,ls={'baseline':'-','time_extend':':','space_refine':'--'}[name]);axes[1,j].plot(xr,100*err,color=color,lw=1.3)
  np.savetxt(root/f'{name}_{key}_comparison.csv',np.c_[xr,yr,ours,err],delimiter=',',header='x_lambda1,paper_vector,ours_aligned,difference',comments='')
for j,key in enumerate(['rho','T']):
 xr,yr=refs[key];axes[0,j].plot(xr,yr,'k-',lw=1.4,label='Chen et al., Fig. 1 S-model',zorder=5)
 axes[0,j].set_title('Density' if key=='rho' else 'Temperature');axes[0,j].set_ylabel('Normalized jump');axes[0,j].set_ylim(-.03,1.05)
 axes[1,j].axhline(0,color='k',lw=.6);axes[1,j].set_ylabel('Difference (pp)');axes[1,j].set_xlabel(r'$x/\lambda_1$ (density-midpoint aligned)')
 for ax in axes[:,j]:ax.set_xlim(-20,15);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9,loc='upper left')
fig.suptitle('Mach 8 Shakhov comparison: Pr = 2/3, viscosity exponent = 0.81',fontsize=14)
fig.text(.5,.015,'Paper curves extracted from PDF vectors; one shared density shift per run. Stationarity criterion not passed.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(root/'m8_paper_comparison.png',dpi=170);fig.savefig(root/'m8_paper_comparison.svg')
report={'source':'https://arxiv.org/pdf/1304.0865','figure':1,'pdf_page':11,'reference_sha256':hashlib.sha256((root/'reference.pdf').read_bytes()).hexdigest(),'extraction':'Original PDF vector polylines, 133 vertices per curve; linear interpolation. Not author-supplied numerical data.','axis_calibration':{'x_range':[-20,15],'x_width_pdf_points':153,'y0_top':227.448,'y1_top':89.662},'normalization':'(rho-rho1)/(rho2-rho1), (T-T1)/(T2-T1); Rankine-Hugoniot states','rho2':r,'T2':T2,'reference_density_midpoint':xrefmid,'metrics':metrics,'limitations':['Published curve discretization/interpolation uncertainty unquantified; plotted line thickness about 0.5 percentage points.','Single density-midpoint shift also used for temperature; no shape fitting.','All runs stationary=false. Signed-distribution negativity persists.','These are macroscopic S-model comparisons, not validation of fourth moments or of historical BGK runs.']}
(root/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(metrics,indent=2))
