"""Run the pre-registered one-factor-at-a-time sensitivity grid (configs/sensitivity.yaml)
on the tuning set (CEC2022 D=10, disjoint seeds). Labels: LA-KAIE-Full@<factor>=<level>;
the default configuration is LA-KAIE-Full@default."""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.experiment import load_yaml  # noqa: E402


def build_overrides():
    sens = load_yaml("sensitivity.yaml")
    ctrl = load_yaml("controller.yaml")
    ov = {"LA-KAIE-Full@default": {}}
    for fname, f in sens["factors"].items():
        path = f["path"]
        levels = f["levels"]
        items = levels.items() if isinstance(levels, dict) else [(v, v) for v in levels]
        default = ctrl
        for k in path:
            default = default[k]
        for tag, val in items:
            if val == default or (isinstance(val, dict) and all(default.get(k) == v for k, v in val.items())):
                continue
            d = val if isinstance(val, dict) else None
            o = cur = {}
            for k in path[:-1]:
                cur[k] = {}; cur = cur[k]
            if d is not None and len(path) == 1:
                o = {path[0]: d}
            else:
                cur[path[-1]] = val
            ov[f"LA-KAIE-Full@{fname}={tag}"] = o
    return ov, sens["tuning"]


if __name__ == "__main__":
    ov, t = build_overrides()
    f = ROOT / "results" / "raw" / "sensitivity_overrides.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.safe_dump(ov, sort_keys=False))
    print(f"{len(ov)} configurations")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--benchmark", "CEC2022",
           "--campaign", "sensitivity", "--dim", str(t["dimension"]), "--max-fes", str(t["max_fes"]),
           "--runs", str(t["runs"]), "--overrides", str(f), "--instances", *map(str, t["instances"])]
    sys.exit(subprocess.call(cmd))
