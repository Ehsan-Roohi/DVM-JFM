from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).resolve().parent
z=np.load(F/'BGK_M3_regenerated/reductions.npz');x=z['x'];p='recomputed_macros_'
def plateau(y):
 n=max(5,int(.08*len(y)));return y-np.mean(np.r_[y[:n],y[-n:]])
u=z[p+'u'];kernel=np.array([1,4,6,4,1])/16
us=np.convolve(np.pad(u,(2,2),mode='edge'),kernel,mode='valid')
rhs=z[p+'qMd']-z[p+'q']-np.gradient(us,x,edge_order=2)*z[p+'A']
lhs=np.gradient(z[p+'F'],x,edge_order=2)
scale=np.max(np.abs(rhs));assert scale>0
fig,ax=plt.subplots(2,1,figsize=(7,5),layout='constrained')
ax[0].plot(x,plateau(z[p+'R']+z[p+'D']/3),'k-',label='full distribution')
ax[0].plot(x,plateau(z[p+'S_rec']),'C1--',label='integrated heat budget')
ax[0].set_ylabel(r'$S$');ax[0].legend();ax[0].set_title(r'BGK, $M_1=3$, $Pr=1$')
ax[1].plot(x,(lhs-rhs)/scale,'C0-');ax[1].set_ylabel(r'$(\partial_xF_q-Q_q+u_x A)/\max|Q_q-u_xA|$')
ax[1].set_xlabel(r'$x/\lambda_1$')
for a in ax:a.set_xlim(-30,30);a.grid(alpha=.2)
fig.savefig(F/'corrected_heat_budget_figure2.pdf');fig.savefig(F/'corrected_heat_budget_figure2.png',dpi=180)
np.savez(F/'corrected_heat_budget_figure2_data.npz',x=x,S=plateau(z[p+'R']+z[p+'D']/3),S_budget=plateau(z[p+'S_rec']),flux_derivative=lhs,heat_RHS=rhs,residual_scale=scale)
