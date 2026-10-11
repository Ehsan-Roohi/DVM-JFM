"""Apply the v24 line-only Figure 7 and profile-only Figure 8 after legacy plotting."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdfplumber
from pypdf import PdfReader,PdfWriter
root=Path(__file__).resolve().parent;inp=root/'inputs';out=root.parent/'figures'
p=pdfplumber.open(inp/'original_fig4_sparse_delta_robustness.pdf').pages[0]
noise=np.array([(143.633895-r['top'])/(143.633895-119.896607) for r in p.rects if r['x0']>200 and r['x1']<328 and r['height']>80 and isinstance(r['non_stroking_color'],tuple)]).reshape(3,3).T
scale=(87.858277-64.759686)/np.log10(2)
rank=np.array([10**(1+(87.858277-r['top'])/scale) for r in p.rects if r['x0']>375 and r['x1']<500 and r['height']>1000 and isinstance(r['non_stroking_color'],tuple)]).reshape(3,3)
with (root/'figure7_archived_table.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['Mach','noise_0_percent','noise_0p5_percent','noise_1_percent','Delta_zero','full_Delta','sparse_Delta'])
 for i,m in enumerate([2,3,5]):w.writerow([m]+[f'{z:.2f}' for z in np.r_[noise[i],rank[i]]])
plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'stix','font.size':15,'axes.labelsize':18,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.5,3.8));fig.subplots_adjust(left=.13,right=.97,bottom=.2,top=.8)
C=['#0072B2','#D55E00','#009E73'];curves=[c for c in p.curves if len(c['pts'])==4 and c['x0']<50 and c['x1']>160]
for c,col,m,mark in zip(curves,C,[2,3,5],['o','s','^']):
 pts=np.array(c['pts']);ax.plot(15+(pts[:,0]-60.326807)/6.0551392,(143.633895-pts[:,1])/5.0158148,marker=mark,color=col,lw=2,label=f'Mach {m}')
for ln,col in zip([l for l in p.lines if l['x0']<37 and l['x1']>169 and isinstance(l['stroking_color'],tuple)],C):ax.axhline((143.633895-ln['top'])/5.0158148,color=col,lw=1.5,ls=':')
ax.set(xlabel=r'Number of interior $\Delta$ probes',ylabel=r'$\varepsilon_R$ (%)');ax.grid(alpha=.18);fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
fig.savefig(out/'fig4_sparse_delta_robustness.pdf',bbox_inches='tight');plt.close(fig)
reader=PdfReader(inp/'original_A4_v23.pdf');page=reader.pages[0];height=float(page.mediabox.top);width=float(page.mediabox.right)
page.cropbox.lower_left=(0,height*.435);page.cropbox.upper_right=(width,height);page.mediabox.lower_left=(0,height*.435);page.mediabox.upper_right=(width,height)
writer=PdfWriter();writer.add_page(page)
with (out/'A4_sparse_core_gap.pdf').open('wb') as f:writer.write(f)
met=json.loads((inp/'reviewer_metrics.json').read_text())['sparse_controlled_audit']
with (root/'figure8_probe_table.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['layout','total_probes','active_probes','Delta_error_percent','R_error_percent'])
 for r in met:w.writerow([r['name'],r['sensors'],r['core_sensors'],r['Delta_error_percent'],r['R_error_percent']])
print('Applied v24 figures and regenerated both table CSV inputs. Original vector/metric provenance retained.')
