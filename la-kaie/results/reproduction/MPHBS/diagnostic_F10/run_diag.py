"""Stage 0 diagnostic (no code change): F10 escape rate of our MPHBS B-C1 vs our
random-action mirror, n1_unit seeds (70000000 + 1000*10 + run), 30 runs each."""
import sys, json, multiprocessing as mp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
import numpy as np
from lakaie.experiment import run_single, load_yaml
cfg = load_yaml("cec2022.yaml")["methods"]["MPHBS"]
def task(label, c, r):
    return {"label": label, "family": "MPHBS", "cfg": c, "benchmark": "CEC2022", "instance": 10, "run": r,
            "campaign": "n1_diag_F10", "seed": 70000000 + 10000 + r, "max_fes": 200000, "N": 30, "dim": 20,
            "n_checkpoints": 50, "out_root": str(Path(__file__).parent / "raw")}
T = [task("BC1", cfg, r) for r in range(1, 31)] + [task("BC1_random", {**cfg, "random_actions": True}, r) for r in range(1, 31)]
if __name__ == "__main__":
    with mp.get_context("fork").Pool(4) as p:
        list(p.imap_unordered(run_single, T))
    out = {}
    for lab in ("BC1", "BC1_random"):
        v = [json.load(open(f))["best_fitness"] for f in (Path(__file__).parent / "raw/n1_diag_F10/CEC2022" / lab).glob("inst10/*.json")]
        out[lab] = {"n": len(v), "escape_lt50": int(np.sum(np.array(v) < 50)), "median": float(np.median(v))}
    (Path(__file__).parent / "summary.json").write_text(json.dumps(out, indent=2)); print(out)
