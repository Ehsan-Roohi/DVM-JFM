"""Manuscript M2 figure: raw 2000-ppc DSMC and matched corrected models."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "M2_raw"
PATHS = {
    "Bird DSMC 2000 ppc, raw": EVIDENCE / "DSMC_2000ppc_M2.npz",
    "Shakhov DVM snapshot": EVIDENCE / "DVM_Shakhov_M2.npz",
    "Corrected Shakhov R26": EVIDENCE / "R26_accepted_Shakhov_M2.npz",
}
FIELDS = ("rho", "T", "q", "R", "Delta", "S")
LABELS = (r"$\rho$", r"$T$", r"$q_x$", r"$R^{\rm cl}_{xx}$", r"$\Delta$", r"$S$")
MIDPOINT = (1 + 16 / 7) / 2


def load(name, path):
    with np.load(path, allow_pickle=False) as data:
        d = {k: data[k] for k in data.files}
    if name == "Shakhov DVM snapshot":
        d.update({str(k): d["fields"][:, i] for i, k in enumerate(d["columns"])})
    if "theta" in d:
        d["T"] = d["theta"]
    x, y = d["x"], d["rho"] - MIDPOINT
    idx = np.flatnonzero(y[:-1] * y[1:] <= 0)
    assert len(idx) == 1
    i = idx[0]
    center = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    d["xc"] = x - center
    assert np.max(np.abs(d["S"] - d["R"] - d["Delta"] / 3)) < 1e-9
    return d


data = {name: load(name, path) for name, path in PATHS.items()}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.9), sharex=True, constrained_layout=True)
styles = {
    "Bird DSMC 2000 ppc, raw": dict(color="#555555", lw=0.8, alpha=0.75),
    "Shakhov DVM snapshot": dict(color="#0072B2", lw=1.8, ls="--"),
    "Corrected Shakhov R26": dict(color="#D55E00", lw=1.8),
}
for ax, field, label in zip(axes.flat, FIELDS, LABELS):
    for name, d in data.items():
        ax.plot(d["xc"], d[field], label=name, **styles[name])
    ax.set_xlim(-15, 15)
    ax.set_ylabel(label)
    ax.set_xlabel(r"$(x-x_{1/2})/\lambda_1$")
    ax.grid(alpha=0.2)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
out = HERE.parent / "figures/M2_DSMC2000_cross_method"
fig.savefig(out.with_suffix(".png"), dpi=250)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(out.name + ".png")
