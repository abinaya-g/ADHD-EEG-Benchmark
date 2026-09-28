"""Apply the pre-registered selection rule (configs/sensitivity.yaml) to the
sensitivity campaign and write configs/selected_parameters.yaml plus
results/tables/sensitivity/*. No other tuning is performed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.stats import load_campaign, mean_matrix, rank_matrix  # noqa: E402
from lakaie.experiment import load_yaml  # noqa: E402
from scripts.run_sensitivity import build_overrides  # noqa: E402

sens = load_yaml("sensitivity.yaml")
ov, tuning = build_overrides()
df = load_campaign(ROOT / "results" / "raw", "sensitivity", "CEC2022")
df = df[df.status == "ok"]
out = ROOT / "results" / "tables" / "sensitivity"; out.mkdir(parents=True, exist_ok=True)
margin = tuning["rank_margin"]
selected, rows = {}, []
for fname in sens["factors"]:
    labels = ["LA-KAIE-Full@default"] + [l for l in ov if l.startswith(f"LA-KAIE-Full@{fname}=")]
    labels = [l for l in labels if l in set(df.method)]
    R = rank_matrix(mean_matrix(df, labels))
    avg = R.mean()
    best = avg.idxmin()
    chosen = "LA-KAIE-Full@default"
    if best != chosen and avg[chosen] - avg[best] >= margin:
        chosen = best
        selected_ov = ov[best]
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict):
                    merge(a.setdefault(k, {}), v)
                else:
                    a[k] = v
        merge(selected, selected_ov)
    for l in labels:
        rows.append({"factor": fname, "level": l.split("@")[1], "avg_rank": avg[l],
                     "mean_error_geomean": float(np.exp(np.mean(np.log(np.maximum(mean_matrix(df, [l])[l], 1e-8))))),
                     "selected": l == chosen})
T = pd.DataFrame(rows)
T.to_csv(out / "sensitivity_ofat.csv", index=False)
(out / "sensitivity_ofat.tex").write_text(T.to_latex(index=False, float_format=lambda v: f"{v:.3g}"))
(ROOT / "configs" / "selected_parameters.yaml").write_text(
    "# Written by scripts/select_parameters.py (pre-registered rule, rank margin "
    f"{margin}). Empty = defaults kept.\n" + yaml.safe_dump(selected or {}, sort_keys=False))
print(T.to_string(index=False)); print("selected overrides:", selected)
