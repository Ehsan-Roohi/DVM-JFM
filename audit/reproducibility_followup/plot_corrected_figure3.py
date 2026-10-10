from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).resolve().parent
def plateau(y):
    n=max(5,int(.08*len(y)));return y-np.mean(np.r_[y[:n],y[-n:]])
figplot,axes=plt.subplots(3,3,figsize=(11,8),constrained_layout=True)
for i,m in enumerate((2,3,5)):
    d=F/f'BGK_M{m}_regenerated';z=np.load(d/'reductions.npz');s=np.load(d/'scalar_window7_reconciliation.npz')
    x=z['x'];p='recomputed_macros_';D=plateau(z[p+'D']);R=plateau(z[p+'R']);S=plateau(z[p+'S_rec']);Db=plateau(s[p+'D_rec_window7'])
    panels=[(D,Db,'Scalar excess'),(R,S,'Heat channel alone'),(R,S-Db/3,'Corrected two budgets')]
    for j,(truth,recovered,title) in enumerate(panels):
        ax=axes[i,j];ax.plot(x,truth,'k-',label='full distribution');ax.plot(x,recovered,'C1--',label='budget')
        
        if j==2:ax.plot(x,S-D/3,'C0:',label=r'full-$\Delta$ reference')
        ax.set_xlim(-{2:20,3:30,5:40}[m],{2:20,3:30,5:40}[m]);ax.grid(alpha=.2);ax.set_xlabel(r'$x/\lambda_1$');ax.set_title(f'M{m}: {title}')
        if i==0:ax.legend(fontsize=8)
figplot.savefig(F/'corrected_two_budget_audit.pdf');figplot.savefig(F/'corrected_two_budget_audit.png',dpi=180);plt.close(figplot)
