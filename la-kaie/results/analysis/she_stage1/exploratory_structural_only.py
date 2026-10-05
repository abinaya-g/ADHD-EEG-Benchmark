"""POST-HOC EXPLORATORY (not pre-registered; does not affect the Stage 1 gate).
Recompute the evidence over the structural hypotheses {h1,h2,h3} only, using only
non-h0 records of the SHE-Uniform runs, from the logged prequential losses.
Approximation: the logged losses come from models that were also fitted on h0
records; refitting without them would change the losses."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from lakaie.she.evidence import posterior
RAW = ROOT / "results/raw/she_diag/SHE-Uniform"
out = {}
for f in (1, 2, 3):
    for D in (10, 20):
        con, pib = [], []
        for jp in sorted((RAW / f"S{f}_D{D}").glob("run*.json")):
            rec = json.loads(jp.read_text()); z = np.load(jp.with_suffix(".npz"))
            m = z["r_gen"] != 0
            ell, w, fe = z["r_loss"][m].astype(float), z["r_w"][m], z["r_fe"][m]
            L = np.zeros(4); P = np.empty((ell.shape[0], 4))
            for n in range(ell.shape[0]):
                P[n] = posterior(L, (1, 2, 3), 1.0, 0.05); L = 0.995 * L + w[n] * np.nan_to_num(ell[n])
            pb = P[fe > 0.1 * rec["max_fe"]].mean(0); pib.append(pb)
            c = {1: pb[1] - pb[3], 2: pb[2] - max(pb[1], pb[3]), 3: pb[3] - max(pb[1], pb[2])}[f]
            con.append(c)
        con = np.array(con)
        p = float(wilcoxon(con, alternative="greater").pvalue) if np.any(con != 0) else 1.0
        out[f"S{f}_D{D}"] = {"median_contrast": float(np.median(con)), "p_one_sided_uncorrected": p,
                             "pibar_mean_h1h2h3": np.mean(pib, 0)[1:].round(3).tolist()}
(Path(__file__).parent / "exploratory_structural_only.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
