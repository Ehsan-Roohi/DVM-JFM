"""Recompute M3 cross-method differences from immutable raw profiles."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE / "M3_raw"
FILES = {
    "DSMC_2000ppc": INPUT / "DSMC_2000ppc_M3.npz",
    "Shakhov_DVM": INPUT / "DVM_Shakhov_M3.npz",
    "Corrected_Shakhov_R26": INPUT / "R26_accepted_Shakhov_M3.npz",
}
FIELDS = ("rho", "T", "q", "R", "Delta", "S")
EXPECTED_R26_SHA = "1b87004303b2d3a5e9c2fc377b2e7eca46e2151a24570b4119237337306b8768"
MIDPOINT = 2.0  # rho1=1, rho2=3 at M1=3 for gamma=5/3


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    with np.load(path, allow_pickle=False) as src:
        d = {key: src[key] for key in src.files}
    if name == "Shakhov_DVM":
        d.update({str(key): d["fields"][:, j] for j, key in enumerate(d["columns"])})
    if "theta" in d:
        d["T"] = d["theta"]
    for field in ("x",) + FIELDS:
        arr = np.asarray(d[field], dtype=np.float64)
        assert arr.ndim == 1 and arr.size == d["x"].size and np.isfinite(arr).all(), (name, field)
        d[field] = arr
    assert np.all(np.diff(d["x"]) > 0), name
    assert np.min(d["rho"]) > 0 and np.min(d["T"]) > 0, name
    assert np.max(np.abs(d["S"] - d["R"] - d["Delta"] / 3)) < 1e-9, name
    y = d["rho"] - MIDPOINT
    idx = np.flatnonzero(y[:-1] * y[1:] <= 0)
    assert idx.size == 1, (name, idx.size)
    j = int(idx[0])
    d["center"] = float(d["x"][j] - y[j] * (d["x"][j + 1] - d["x"][j]) / (y[j + 1] - y[j]))
    return d


def compare(points):
    x = np.linspace(-15, 15, points)
    aligned = {name: {field: np.interp(x, d["x"] - d["center"], d[field])
                      for field in FIELDS} for name, d in DATA.items()}
    dsmc = aligned["DSMC_2000ppc"]
    return {
        name: {
            field: float(100 * np.linalg.norm(values[field] - dsmc[field]) / np.linalg.norm(dsmc[field]))
            for field in FIELDS
        }
        for name, values in aligned.items() if name != "DSMC_2000ppc"
    }, aligned


assert sha256(FILES["Corrected_Shakhov_R26"]) == EXPECTED_R26_SHA
DATA = {name: load(name, path) for name, path in FILES.items()}
METRICS, ALIGNED = compare(1201)
METRICS_2401, _ = compare(2401)
channel = {}
ref = ALIGNED["DSMC_2000ppc"]
for name in ("Shakhov_DVM", "Corrected_Shakhov_R26"):
    e_r = ALIGNED[name]["R"] - ref["R"]
    e_d3 = (ALIGNED[name]["Delta"] - ref["Delta"]) / 3
    e_s = ALIGNED[name]["S"] - ref["S"]
    nr, nd, ns = (np.linalg.norm(a) for a in (e_r, e_d3, e_s))
    scale = np.linalg.norm(ref["S"])
    assert np.max(np.abs(e_s - e_r - e_d3)) < 1e-10
    channel[name] = {
        "error_R_percent_common_S_denominator": float(100 * nr / scale),
        "error_Delta_over_3_percent_common_S_denominator": float(100 * nd / scale),
        "error_S_percent_common_S_denominator": float(100 * ns / scale),
        "signed_cross_correlation": float(np.dot(e_r, e_d3) / (nr * nd)),
        "cancellation_ratio": float(ns / (nr + nd)),
    }
result = {
    "method": "100*||model-DSMC2000||2/||DSMC2000||2; x in [-15,15] lambda1; each profile translated by its linearly interpolated RH density midpoint; no width fit/smoothing",
    "M": 3,
    "rho_midpoint": MIDPOINT,
    "fields": FIELDS,
    "source_sha256": {name: sha256(path) for name, path in FILES.items()},
    "midpoint_x": {name: d["center"] for name, d in DATA.items()},
    "L2_percent_DSMC_denominator_1201": METRICS,
    "L2_percent_DSMC_denominator_2401_check": METRICS_2401,
    "channel_signed_error_common_DSMC_S_denominator": channel,
    "notes": [
        "R26 is accepted corrected Shakhov p4/q8 profile; it is not the historical pre-STF or Maxwell profile.",
        "All metrics use unsmoothed raw profiles; the 1201 interpolated points are not independent samples.",
        "DSMC is one-seed finite-time data; DVM is a nonstationary saved snapshot. Pairwise differences are descriptive, not certified model errors.",
    ],
}
(HERE / "DSMC2000_M3_corrected_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
with (HERE / "DSMC2000_M3_corrected_comparison.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(("model",) + FIELDS)
    for model, values in METRICS.items():
        writer.writerow((model,) + tuple(values[k] for k in FIELDS))
print(json.dumps(result, indent=2))
