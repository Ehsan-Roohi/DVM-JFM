"""Read-only export of atomic live DVM checkpoints to immutable profiles."""
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys
from datetime import datetime, timezone

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = pathlib.Path(sys.argv[1])
    project = pathlib.Path(sys.argv[2])
    spec = importlib.util.spec_from_file_location("dvm_solver", root / "solver.py")
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    manifest = json.loads((root / "manifest.json").read_text())
    source = sha(root / "solver.py")
    assert source == "9a20d251b6328e1e49c67e53663f4339ea907c2732bcc89ccb7559c9af20cd64"
    assert sha(root / "manifest.json") == "f570d4e07eb352720e736dded7f47fd730d500cf4b780565a3b925354b5ce0e7"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local = root / "exports" / ("live_profiles_" + stamp)
    local.mkdir(parents=True, exist_ok=False)
    report = {"created_utc": stamp, "label": "TRANSIENT_NOT_STATIONARY", "source_sha256": source,
              "manifest_sha256": sha(root / "manifest.json"), "cases": {}}
    for config in manifest["cases"]:
        name = config["name"]
        cp = root / "runs" / name / "checkpoint.npz"
        with np.load(cp, allow_pickle=False) as data:
            state = data["f"].copy()
            signature = str(data["signature"])
            step = int(data["step"])
            stable = int(data["stable"])
            prev = data["prev"].copy()
        expected = hashlib.sha256((source + json.dumps(config, sort_keys=True)).encode()).hexdigest()
        assert signature == expected
        x, v, w, initial, left, right = solver.setup(config)
        assert state.shape == initial.shape and np.isfinite(state).all()
        fields = solver.moments(state, v, w)
        assert fields.shape == (config["nx"], 8) and np.isfinite(fields).all()
        assert np.all(np.diff(x) > 0)
        assert np.all(fields[:, 0] > 0) and np.all(fields[:, 2] > 0)
        assert np.max(abs(fields[:, 7] - fields[:, 5] - fields[:, 6] / 3)) < 1e-12
        dt = config["CFL"] * 2 * config["L"] / config["nx"] / config["vmax"]
        drift = None
        if prev.size:
            assert prev.shape == fields.shape
            drift = float(np.max(np.max(abs(fields - prev), axis=0) /
                                 np.maximum(np.max(abs(fields), axis=0), 1e-8)))
        output = local / (name + ".npz")
        np.savez_compressed(output, x=x, fields=fields,
                            columns=np.array(["rho", "u", "T", "q", "sigma", "R", "Delta", "S"]))
        report["cases"][name] = {"mach": config["mach"], "step": step, "time": step * dt,
                                  "stable_count": stable, "drift_from_saved_previous": drift,
                                  "stationary_certified": False, "finished": False,
                                  "source_checkpoint": str(cp), "source_checkpoint_sha256": sha(cp),
                                  "profile_sha256": sha(output), "shape": list(fields.shape),
                                  "min_rho": float(fields[:, 0].min()),
                                  "min_T": float(fields[:, 2].min()), "config": config}
    receipt = local / "receipt.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    project.mkdir(parents=True, exist_ok=True)
    destination = project / local.name
    destination.mkdir(exist_ok=False)
    for path in local.iterdir():
        target = destination / path.name
        shutil.copyfile(path, target)
        assert sha(path) == sha(target)
    print(json.dumps({"scratch": str(local), "project": str(destination), "receipt": report}, separators=(",", ":")))


if __name__ == "__main__":
    main()
