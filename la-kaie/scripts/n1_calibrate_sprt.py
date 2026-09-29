"""T-SPRT calibration (spec §7.2; Gate 8). Empirical behaviour only - no
classical Wald guarantee is claimed.

Level 1 - semi-synthetic replay (0 FE): the logged, online-estimated p0
  sequences of the A7 smoke runs are replayed through the unchanged rule with
  simulated outcomes y ~ Bernoulli(sigma(logit p0 + offset + u_t)),
  offset in {0 (controlled no-transfer), delta (controlled positive transfer)},
  u_t ~ N(0, sigma_u^2) shared within iteration t, sigma_u in {0, 0.5}.
  Data-source note: spec §7.2 says "from A0 runs", but A0 has no exchange
  candidates (hence no exchange step lengths / p0); the A7 smoke logs are used.
Level 2 - pipeline harness: arms H_null (native-equivalent candidates) and
  H_pos (donor = known optimum) on S1, S3, D in {10, 20}, 5 runs
  (campaign n1_calib; run with scripts/n1_run.py --campaign n1_calib).

Reported at both levels: false activation rate (null), false suppression rate
(positive), stopping-time distribution, truncation fraction, Monte Carlo SE.
"Grossly off" (STOP): Level 1, sigma_u = 0, false activation or false
suppression > 3 x nominal (0.15).

  python scripts/n1_calibrate_sprt.py [--reps 20]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from n1.config import N1Config  # noqa: E402
from n1.sprt import ACTIVE, INFORMATIVE, UNINFORMATIVE, SPRTStyleRule  # noqa: E402

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "n1_calibration"


def _se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-12) / n) if n else float("nan")


def stopping_stats(ns):
    ns = np.asarray(ns, float)
    if ns.size == 0:
        return {"n_decisions": 0}
    return {"n_decisions": int(ns.size), "median": float(np.median(ns)),
            "q25": float(np.quantile(ns, 0.25)), "q75": float(np.quantile(ns, 0.75)),
            "p95": float(np.quantile(ns, 0.95)), "max": float(ns.max())}


def replay(seq_t, seq_p0, offset, sigma_u, rng, cfg):
    rule = SPRTStyleRule(cfg)
    u = {}
    n_supp = 0
    for t, p0 in zip(seq_t, seq_p0):
        if t not in u:
            u[t] = rng.normal(0.0, sigma_u) if sigma_u > 0 else 0.0
        z = math.log(p0 / (1 - p0)) + offset + u[t]
        y = int(rng.random() < 1.0 / (1.0 + math.exp(-z)))
        rule.update(p0, y, t)
        n_supp += int(rule.state != ACTIVE)
    return rule.decisions, rule.state, n_supp / max(1, len(seq_p0))


def level1(reps, cfg):
    runs = []
    for p in sorted((RAW / "n1_smoke" / "A7").glob("*/run*.npz")):
        z = np.load(p, allow_pickle=False)
        if "x_p0" in z and z["x_p0"].size:
            runs.append((p.parent.name, z["x_t"], z["x_p0"]))
    rng = np.random.default_rng(83000001)
    res = {}
    for cond, offset in (("null", 0.0), ("positive", cfg.delta)):
        for su in (0.0, 0.5):
            dec, ns, trunc, final_active, supp = [], [], 0, [], []
            for _, t, p0 in runs:
                for _ in range(reps):
                    d, state, fsup = replay(t, p0, offset, su, rng, cfg)
                    dec += [x["decision"] for x in d]
                    ns += [x["n"] for x in d]
                    trunc += sum(x["truncated"] for x in d)
                    final_active.append(state == ACTIVE)
                    supp.append(fsup)
            n = len(dec)
            key = f"{cond}_sigma{su}"
            if cond == "null":
                rate = float(np.mean([x == INFORMATIVE for x in dec])) if n else float("nan")
                res[key] = {"false_activation_rate": rate, "se": _se(rate, n),
                            "frac_sequences_ending_active": float(np.mean(final_active))}
            else:
                rate = float(np.mean([x == UNINFORMATIVE for x in dec])) if n else float("nan")
                res[key] = {"false_suppression_rate": rate, "se": _se(rate, n),
                            "frac_outcomes_while_suppressed": float(np.mean(supp))}
            res[key].update({"stopping": stopping_stats(ns), "truncation_fraction": trunc / n if n else float("nan"),
                             "n_sequences": len(runs) * reps})
    return res, len(runs)


def level2():
    res = {}
    for arm, cond in (("H_null", "null"), ("H_pos", "positive")):
        dec, ns, trunc, final_active, supp_frac = [], [], 0, [], []
        files = sorted((RAW / "n1_calib" / arm).glob("*/run*.json"))
        for p in files:
            r = json.loads(p.read_text())
            s = r["summary"]
            ds = s.get("decisions", [])
            dec += [x["decision"] for x in ds]
            ns += [x["n"] for x in ds]
            trunc += sum(bool(x["truncated"]) for x in ds)
            final_active.append(s.get("final_state") == ACTIVE)
            z = np.load(p.with_suffix(".npz"))
            st = z["iter_state"] if "iter_state" in z else np.array([])
            supp_frac.append(float(np.mean(st == 0.0)) if st.size else float("nan"))
        n = len(dec)
        if cond == "null":
            rate = float(np.mean([x == INFORMATIVE for x in dec])) if n else float("nan")
            res[cond] = {"false_activation_rate": rate, "se": _se(rate, n),
                         "frac_runs_ending_active": float(np.mean(final_active)) if final_active else float("nan")}
        else:
            rate = float(np.mean([x == UNINFORMATIVE for x in dec])) if n else float("nan")
            res[cond] = {"false_suppression_rate": rate, "se": _se(rate, n),
                         "frac_iterations_suppressed": float(np.nanmean(supp_frac)) if supp_frac else float("nan")}
        res[cond].update({"stopping": stopping_stats(ns), "truncation_fraction": trunc / n if n else float("nan"),
                          "n_runs": len(files)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    cfg = N1Config()
    OUT.mkdir(parents=True, exist_ok=True)
    L1, n_seq = level1(args.reps, cfg)
    L2 = level2()
    nominal = cfg.alpha
    fa0 = L1.get("null_sigma0.0", {}).get("false_activation_rate", float("nan"))
    fs0 = L1.get("positive_sigma0.0", {}).get("false_suppression_rate", float("nan"))
    grossly_off = bool((fa0 > 3 * nominal) or (fs0 > 3 * cfg.beta) or not np.isfinite(fa0) or not np.isfinite(fs0))
    out = {"level1": L1, "level1_source_runs": n_seq, "level2": L2, "nominal_alpha": cfg.alpha,
           "nominal_beta": cfg.beta, "grossly_off": grossly_off,
           "criterion": "Level 1 sigma_u=0: false activation or false suppression > 3 x nominal -> STOP"}
    (OUT / "calibration.json").write_text(json.dumps(out, indent=2))

    def fmt(d, key):
        s = d.get("stopping", {})
        return (f"| {d.get(key, float('nan')):.4f} ± {d.get('se', float('nan')):.4f} | "
                f"{s.get('n_decisions', 0)} | {s.get('median', float('nan')):.0f} "
                f"[{s.get('q25', float('nan')):.0f}, {s.get('q75', float('nan')):.0f}] | "
                f"{s.get('p95', float('nan')):.0f} | {d.get('truncation_fraction', float('nan')):.3f} |")
    L = ["# T-SPRT calibration report (spec §7.2)\n",
         "This report gives the **empirical behaviour** of the SPRT-style prequential sequential "
         "likelihood-ratio decision rule. **No classical Wald type-I/type-II guarantee is claimed.** "
         f"The nominal α = {cfg.alpha} and β = {cfg.beta} only set the boundaries.\n",
         "## Level 1: semi-synthetic replay (0 FE)\n",
         f"- Source: the logged, online-estimated p0 sequences of {n_seq} A7 smoke runs.",
         "  - Spec §7.2 names A0 as the source, but A0 has no exchange candidates. This is a "
         "documented deviation.",
         f"- Outcomes are simulated. Each p0 sequence is replayed {args.reps} times per condition.",
         "- The ± value is the binomial Monte Carlo SE over decisions. Decisions within a sequence "
         "are dependent, so this SE is optimistic.\n",
         "| condition | rate ± SE | decisions | stopping n: median [IQR] | p95 | truncation |",
         "|---|---|---|---|---|---|"]
    for key, d in L1.items():
        L.append(f"| {key} ({'false activation' if key.startswith('null') else 'false suppression'}) "
                 + fmt(d, "false_activation_rate" if key.startswith("null") else "false_suppression_rate"))
    L += ["", "Additional Level 1 quantities:\n"]
    for key, d in L1.items():
        extra = {k: v for k, v in d.items() if k.startswith("frac")}
        L.append(f"- {key}: {extra}")
    L += ["", "## Level 2: pipeline harness (test fixtures, not N1 variants)\n",
          "- Arms: H_null (native-equivalent exchange candidates) and H_pos (donor = known optimum).",
          "- Problems: S1 and S3, D ∈ {10, 20}, 5 runs each.\n",
          "| condition | rate ± SE | decisions | stopping n: median [IQR] | p95 | truncation |",
          "|---|---|---|---|---|---|"]
    for cond, d in L2.items():
        L.append(f"| {cond} ({'false activation' if cond == 'null' else 'false suppression'}) "
                 + fmt(d, "false_activation_rate" if cond == "null" else "false_suppression_rate"))
    L += ["", "Additional Level 2 quantities:\n"]
    for cond, d in L2.items():
        L.append(f"- {cond}: {({k: v for k, v in d.items() if k.startswith('frac') or k == 'n_runs'})}")
    L += ["", "## Gross-miscalibration check\n",
          f"- The criterion is Level 1 with σ_u = 0: false activation {fa0:.4f} and false "
          f"suppression {fs0:.4f}, each against the limit 3 × 0.05 = 0.15.",
          f"- **Result: {'GROSSLY OFF — STOP' if grossly_off else 'not grossly off'}.**",
          "- Caveat on the Level 2 harness: exchange receivers are chosen by tournament, so they "
          "are fitter than average.",
          "  - Under the null, a native-equivalent step from a fitter parent succeeds *less* often "
          "than the average native attempt at the same step length.",
          "  - The harness null is therefore conservative toward suppression.",
          "  - This is reported as-is.", ""]
    (OUT / "T_SPRT_CALIBRATION_REPORT.md").write_text("\n".join(L))
    print("\n".join(L))
    sys.exit(2 if grossly_off else 0)


if __name__ == "__main__":
    main()
