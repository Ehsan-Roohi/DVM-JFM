"""Bounded archival CSV verification. No kinetic solver or fullstate processing.

Usage: python verify_figure5_csv.py PROFILE_CSV METRIC_CSV OUTPUT_JSON
Uses the stored mask and the archived 8-percent endpoint convention.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

profiles, metrics, output = map(Path, sys.argv[1:4])
rows = list(csv.DictReader(profiles.open(encoding="utf-8")))
data = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
metric = next(csv.DictReader(metrics.open(encoding="utf-8")))
n = len(rows)
ne = max(5, int(0.08 * n))
edges = np.r_[np.arange(ne), np.arange(n - ne, n)]
mask = data["active_mask"] > 0
assert np.array_equal(mask, np.abs(data["S"]) > 0.02 * np.max(np.abs(data["S"])))
channels = ["R", "Delta", "S", "S_from_flux", "Delta_from_source", "Delta_target"]
means = {key: float(np.mean(data[key][edges])) for key in channels}
assert max(map(abs, means.values())) < 1e-14
norms = {}
for name, a, b in [
    ("S_flux_relL2_active", "S_from_flux", "S"),
    ("Delta_source_relL2_active", "Delta_from_source", "Delta"),
    ("R_delta0_relL2_active", "R_delta0", "R"),
    ("R_two_channel_relL2_active", "R_two_channel", "R"),
]:
    value = float(np.linalg.norm((data[a] - data[b])[mask]) / np.linalg.norm(data[b][mask]))
    np.testing.assert_allclose(value, float(metric[name]), rtol=1e-13, atol=1e-16)
    norms[name] = value
offset = data["Delta_from_source"] + float(metric["Kn"]) * data["Qdelta_direct"]
assert np.ptp(offset) < 1e-13
np.testing.assert_allclose(data["S"], data["R"] + data["Delta"] / 3, atol=3e-14)
np.testing.assert_allclose(data["R_two_channel"], data["S_from_flux"] - data["Delta_from_source"] / 3, atol=3e-14)
result = {
    "status": "PASS_ARCHIVED_CSV_ONLY",
    "profile_sha256": hashlib.sha256(profiles.read_bytes()).hexdigest(),
    "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
    "points": n, "endpoint_points_each": ne, "active_points": int(mask.sum()),
    "channel_endpoint_means": means, "reproduced_norms": norms,
    "Delta_source_minus_raw_source_offset_mean": float(np.mean(offset)),
    "offset_peak_to_peak": float(np.ptp(offset)),
    "limitations": "No full-distribution regeneration, stationary-shock or nonnegative-target certification; no new physical numbers.",
}
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
