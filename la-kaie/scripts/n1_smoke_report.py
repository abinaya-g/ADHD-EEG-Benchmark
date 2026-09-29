"""Evaluate Gates 1-9 on the mechanism smoke campaign and write
MECHANISM_SMOKE_REPORT.md (spec §16; stage 3a/3b). Thresholds are the
a-priori ones of spec §16 - nothing here is tuned.

Inputs: results/raw/n1_smoke (A0, A0_noshadow, A7), results/raw/n1_calib,
results/n1_calibration/calibration.json, results/n1_smoke/pytest_junit.xml.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from n1.config import N1Config  # noqa: E402
from n1.selectors import evaluate_sources, joint_half_sources  # noqa: E402

RAW = ROOT / "results" / "raw" / "n1_smoke"
CFG = N1Config()


def load(arm):
    out = []
    for p in sorted((RAW / arm).glob("*/run*.json")):
        r = json.loads(p.read_text())
        r["_npz"] = p.with_suffix(".npz")
        out.append(r)
    return out


def ari(a_labels, b_labels):
    from math import comb
    n = len(a_labels)
    ct = Counter(zip(a_labels, b_labels))
    sa, sb = Counter(a_labels), Counter(b_labels)
    idx = sum(comb(v, 2) for v in ct.values())
    ea = sum(comb(v, 2) for v in sa.values())
    eb = sum(comb(v, 2) for v in sb.values())
    exp = ea * eb / comb(n, 2)
    mx = (ea + eb) / 2
    return 1.0 if mx == exp else (idx - exp) / (mx - exp)


def part_labels(partition, D):
    lab = np.zeros(D, int)
    for i, b in enumerate(partition):
        lab[np.asarray(b, int)] = i
    return lab.tolist()


def sel_rows(z, selector):
    m = z["s_selector"] == selector
    return {k[2:]: z[k][m] for k in z.files if k.startswith("s_")}


def junit():
    p = ROOT / "results" / "n1_smoke" / "pytest_junit.xml"
    if not p.exists():
        return None
    root = ET.parse(p).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    names = [c.get("name") for s in suites for c in s.iter("testcase")
             if not list(c.iter("failure")) and not list(c.iter("error"))]
    return tot, names


def main():
    L, gates, stop = [], {}, []
    A0, A0n, A7 = load("A0"), load("A0_noshadow"), load("A7")
    allr = A0 + A0n + A7

    # ---- Gate 1
    j = junit()
    if j is None:
        gates[1] = (False, "pytest junit file missing")
    else:
        tot, _ = j
        ok = tot["failures"] == 0 and tot["errors"] == 0 and tot["tests"] > 0
        gates[1] = (ok, f"{tot['tests']} tests, {tot['failures']} failures, {tot['errors']} errors, "
                        f"{tot['skipped']} skipped (full suite: tests/ incl. tests/n1 and the existing "
                        f"baseline tests = T-MPHBS)")

    # ---- Gate 2: FE accounting
    bad = [r for r in allr if not (r["status"] == "ok" and r["summary"].get("fe_audit_ok")
                                   and r["summary"].get("fe_unused_ok"))]
    n_exp = 2 * 8 * 2 * 5 + 3 * 2 * 5
    ok2 = len(allr) == n_exp and not bad
    fe_rows = []
    for arm, rs in (("A0", A0), ("A0_noshadow", A0n), ("A7", A7)):
        tags = Counter()
        for r in rs:
            tags.update(r["summary"].get("fe_by_tag", {}))
        unused = [r["summary"].get("fe_unused", np.nan) for r in rs]
        fe_rows.append(f"| {arm} | {len(rs)} | {dict(tags)} | {int(np.max(unused)) if rs else '–'} | "
                       f"{sum(1 for r in rs if r['summary'].get('fe_audit_ok'))} |")
    gates[2] = (ok2, f"{len(allr)}/{n_exp} runs; runs failing status/audit/unused-bound: {len(bad)}")

    # ---- Gate 3: shadow neutrality + no hidden FE
    key = lambda r: (r["fid"], r["dimension"], r["run"])        # noqa: E731
    m0 = {key(r): r for r in A0}
    mism = []
    for r in A0n:
        a = m0.get(key(r))
        if a is None:
            mism.append((key(r), "missing A0"))
            continue
        ca, cb = np.load(a["_npz"])["curve"], np.load(r["_npz"])["curve"]
        if not (a["FE"] == r["FE"] and a["final_best"] == r["final_best"] and np.array_equal(ca, cb, equal_nan=True)):
            mism.append((key(r), "trajectory differs"))
    hidden_ok = all(r["summary"].get("n_hidden_checks") == r["summary"].get("n_iter") for r in allr)
    unit3 = j is not None and j[0]["failures"] == 0 and j[0]["errors"] == 0
    gates[3] = (not mism and hidden_ok and len(A0n) == 30 and unit3,
                f"A0 vs A0_noshadow pairs compared: {len(A0n)}; mismatches: {len(mism)}; "
                f"zero-FE section check passed in every iteration of every run: {hidden_ok}")

    # ---- Gate 4: prequential / leakage
    order_bad = win_bad = 0
    oos = []
    for r in A7:
        z = np.load(r["_npz"])
        if "x_fe_p0" in z and not np.all(z["x_fe_eval"] == z["x_fe_p0"] + 1):
            order_bad += 1
        if "x_oos" in z and z["x_oos"].size:
            oos.append(float(np.mean(z["x_oos"])))
    for r in A0 + A7:
        z = np.load(r["_npz"])
        ops, ns = z["na_op"], z["na_n_succ"]
        s = r["summary"]
        if (s["n_window_rows_H"] != int(ns[ops == "HBA"].sum())
                or s["n_window_rows_M"] != int(ns[(ops == "MPA") | (ops == "FAD")].sum())):
            win_bad += 1
    gates[4] = (order_bad == 0 and win_bad == 0 and unit3,
                f"A7 runs with a p0-after-evaluation record: {order_bad}; runs whose window rows differ from "
                f"the native-success count (exchange leakage): {win_bad}; T-PREQ unit tests passed: {unit3}")
    oos_max = max(oos) if oos else float("nan")
    if oos and oos_max > 0.5:
        stop.append(f"R-SPRT-4: out-of-support exchange fraction {oos_max:.2f} > 0.5 in at least one A7 run")

    # ---- selection summaries (A0, JOINT) for Gates 5-7, 9
    def joint_summary(fid, D, selector="JOINT"):
        rows = []
        for r in A0:
            if r["fid"] != fid or r["dimension"] != D:
                continue
            z = np.load(r["_npz"])
            s = sel_rows(z, selector)
            ev = s["evidence"].astype(bool)
            rows.append({"run": r["run"], "ev": ev, "k_cur": s["k_cur"], "k_hat": s["k_hat"],
                         "nH": s["nH"], "nM": s["nM"], "bic": {k: s[f"bic_{k}"] for k in ("HI", "HB", "HR")},
                         "partition": s["partition"], "true": r["true_partition"], "W": r["summary"]["W"]})
        return rows

    def modal_ok(fid, label):
        res = {}
        for D in (10, 20):
            rows = joint_summary(fid, D)
            modal = []
            for x in rows:
                kc = x["k_cur"][x["ev"]]
                modal.append(Counter(kc.tolist()).most_common(1)[0][0] if kc.size else "none")
            res[D] = (sum(m == label for m in modal), len(rows), modal)
        return res

    g5 = modal_ok(1, "HI")
    gates[5] = (all(v[0] >= 4 for v in g5.values()),
                "; ".join(f"D={D}: HI modal in {v[0]}/{v[1]} runs (modal labels {v[2]})" for D, v in g5.items()))
    g6 = modal_ok(2, "HB")
    aris = {}
    for D in (10, 20):
        vals = []
        for x in joint_summary(2, D):
            parts = [p for p, e in zip(x["partition"], x["ev"]) if e and p]
            if parts and x["true"]:
                est = [list(map(int, b.split(","))) for b in parts[-1].split(";")]
                vals.append(ari(part_labels(est, D), part_labels(x["true"], D)))
        aris[D] = vals
    gates[6] = (all(v[0] >= 4 for v in g6.values()),
                "; ".join(f"D={D}: HB modal in {v[0]}/{v[1]} runs (modal labels {v[2]})" for D, v in g6.items())
                + "; ARI of the last joint partition vs truth: "
                + "; ".join(f"D={D}: {np.round(v, 3).tolist()}" for D, v in aris.items()))

    # ---- Gate 7: S3 HR opportunity and detectability
    g7, diag7 = True, []
    for D in (10, 20):
        rows = joint_summary(3, D)
        ev_frac = [float(np.mean(x["ev"])) if x["ev"].size else 0.0 for x in rows]
        opp = float(np.mean(ev_frac)) if ev_frac else 0.0
        kc = np.concatenate([x["k_cur"][x["ev"]] for x in rows]) if rows else np.array([])
        kh = np.concatenate([x["k_hat"][x["ev"]] for x in rows]) if rows else np.array([])
        hr_cur = float(np.mean(kc == "HR")) if kc.size else 0.0
        hr_hat = float(np.mean(kh == "HR")) if kh.size else 0.0
        b = {k: np.concatenate([x["bic"][k][x["ev"]] for x in rows]) if rows else np.array([]) for k in ("HI", "HB", "HR")}
        best_other = np.fmin(b["HI"], np.where(np.isnan(b["HB"]), np.inf, b["HB"])) if b["HI"].size else np.array([])
        dHR = b["HR"] - best_other if b["HR"].size else np.array([])
        a_ok, b_ok = opp >= 0.5, hr_cur >= 0.05
        g7 = g7 and a_ok and b_ok
        freq = dict(Counter(kc.tolist()))
        diag7.append(f"D={D}: (a) fraction of selection epochs with >= W samples per population "
                     f"(JOINT evidence) = {opp:.2f} [{'pass' if a_ok else 'FAIL'}]; (b) HR share of "
                     f"evidence-bearing epochs: k_cur {hr_cur:.3f}, k_hat {hr_hat:.3f} "
                     f"[{'pass' if b_ok else 'FAIL'}]; HR ever selected: {bool(hr_hat > 0)}; "
                     f"k_cur frequencies {freq}; W = {rows[0]['W'] if rows else '–'} per population; "
                     f"median BIC HI/HB/HR = {np.nanmedian(b['HI']):.1f} / "
                     f"{np.nanmedian(b['HB']) if np.isfinite(b['HB']).any() else float('nan'):.1f} / "
                     f"{np.nanmedian(b['HR']):.1f}; ΔBIC(HR − best other) median {np.nanmedian(dHR):.1f}, "
                     f"min {np.nanmin(dHR) if dHR.size else float('nan'):.1f}")
    gates[7] = (g7, " | ".join(diag7))

    # ---- Gate 8: calibration
    cal = ROOT / "results" / "n1_calibration" / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        l2 = c.get("level2", {})
        done = bool(c.get("level1")) and all(v.get("n_runs", 0) > 0 for v in l2.values()) and len(l2) == 2
        gates[8] = (done and not c["grossly_off"],
                    f"Level 1 and Level 2 reported: {done}; grossly off: {c['grossly_off']} "
                    f"(see results/n1_calibration/T_SPRT_CALIBRATION_REPORT.md)")
    else:
        gates[8] = (False, "calibration not run")

    # ---- Gate 9: JOINT vs POOLED operational validity
    dig_bad = ep_tot = j_valid = p_valid = 0
    jh_ok = jh_tot = 0
    for r in A0:
        z = np.load(r["_npz"])
        sj, sp = sel_rows(z, "JOINT"), sel_rows(z, "POOLED")
        ev = sj["evidence"].astype(bool)
        ep_tot += int(ev.sum())
        dig_bad += int(np.sum(ev & (sj["digest"] != sp["digest"])))
        j_valid += int(np.sum(ev & (sj["k_hat"] != "") & ~sj["numerical_failure"].astype(bool)))
        p_valid += int(np.sum(ev & (sp["k_hat"] != "") & ~sp["numerical_failure"].astype(bool)))
        if r["fid"] in (1, 2, 3) and "streamH_rows" in z:
            W = r["summary"]["W"]
            RH, RM = z["streamH_rows"].astype(float), z["streamM_rows"].astype(float)
            for nH, nM in list(zip(z["ep_nH"], z["ep_nM"]))[::10]:
                if nH >= 2 * W and nM >= 2 * W:
                    jh_tot += 1
                    wH, wM = RH[:nH][-2 * W:], RM[:nM][-2 * W:]
                    src = joint_half_sources(wH[W:], wH[:W], wM[W:], wM[:W])
                    res = evaluate_sources("JOINT-HALF", src, "stouffer", CFG)
                    jh_ok += int(res.evidence and sum(res.n.values()) == W)
    ok9 = (dig_bad == 0 and ep_tot > 0 and j_valid >= 0.9 * ep_tot and p_valid >= 0.9 * ep_tot
           and jh_tot > 0 and jh_ok == jh_tot and unit3)
    gates[9] = (ok9, f"evidence-bearing JOINT epochs: {ep_tot}; epochs with differing JOINT/POOLED "
                     f"observation digests: {dig_bad}; valid selections JOINT {j_valid}, POOLED {p_valid}; "
                     f"offline JOINT-HALF recomputed at {jh_ok}/{jh_tot} sampled epochs (n = W each)")

    # ---- descriptive tables
    desc = ["| function | D | selector | evidence epochs | k_cur shares (HI/HB/HR) |", "|---|---|---|---|---|"]
    for fid in range(1, 9):
        for D in (10, 20):
            for sel in ("HBA", "MPA", "JOINT", "POOLED", "DECOUPLED"):
                rows = joint_summary(fid, D, sel)
                kc = np.concatenate([x["k_cur"][x["ev"]] for x in rows]) if rows else np.array([])
                sh = "/".join(f"{np.mean(kc == k):.2f}" if kc.size else "–" for k in ("HI", "HB", "HR"))
                desc.append(f"| S{fid} | {D} | {sel} | {kc.size} | {sh} |")
    a7 = ["| function | D | final best (median) | ACTIVE-iteration share | decisions / run | "
          "truncated share | out-of-support share | fallbacks / run |", "|---|---|---|---|---|---|---|---|"]
    for fid in range(1, 9):
        for D in (10, 20):
            rs = [r for r in A7 if r["fid"] == fid and r["dimension"] == D]
            if not rs:
                continue
            act, oo = [], []
            for r in rs:
                z = np.load(r["_npz"])
                act.append(float(np.mean(z["iter_state"])) if "iter_state" in z else np.nan)
                oo.append(float(np.mean(z["x_oos"])) if "x_oos" in z and z["x_oos"].size else np.nan)
            dec = [len(r["summary"]["decisions"]) for r in rs]
            tr = [np.mean([d["truncated"] for d in r["summary"]["decisions"]]) if r["summary"]["decisions"] else np.nan
                  for r in rs]
            a7.append(f"| S{fid} | {D} | {np.median([r['final_best'] for r in rs]):.3e} | {np.nanmean(act):.2f} | "
                      f"{np.mean(dec):.1f} | {np.nanmean(tr):.2f} | {np.nanmean(oo):.2f} | "
                      f"{np.mean([r['summary']['n_fallback'] for r in rs]):.1f} |")

    # ---- report
    names = {1: "unit tests", 2: "FE accounting", 3: "shadow neutrality / no hidden FE", 4: "prequential leakage",
             5: "S1 / HI behaviour", 6: "S2 / HB behaviour", 7: "S3 / HR opportunity and detectability",
             8: "SPRT calibration", 9: "JOINT/POOLED operational validity"}
    passed = [g for g, (ok, _) in gates.items() if ok]
    failed = [g for g, (ok, _) in gates.items() if not ok]
    L += ["# Mechanism smoke report (Stage 3a/3b)\n",
          "Generated by `scripts/n1_smoke_report.py` from the raw runs. Gate thresholds are the "
          "a-priori ones of spec §16. **Nothing was tuned.**\n",
          "- Smoke campaign `n1_smoke`: arms A0 and A7 on S1–S8 × D ∈ {10, 20} × 5 runs, plus "
          "A0_noshadow on S1–S3 (amendment A-5).",
          "- Calibration campaign: `n1_calib`.\n",
          "## Gate summary\n", "| gate | name | result | evidence |", "|---|---|---|---|"]
    for g in range(1, 10):
        ok, ev = gates[g]
        L.append(f"| {g} | {names[g]} | **{'PASS' if ok else 'FAIL'}** | {ev} |")
    L += ["", f"Passed: {passed}. Failed: {failed}.", ""]
    if stop:
        L += ["Additional hard-stop conditions triggered:", *[f"- {s}" for s in stop], ""]
    L += ["## FE audit (all smoke runs)\n", "| arm | runs | FE by phase (summed) | max unused FE | audit ok |",
          "|---|---|---|---|---|", *fe_rows, "",
          "## Selection shares in A0 (passive shadows; descriptive, not confirmatory)\n", *desc, "",
          "## A7 (full method) mechanism summary (descriptive)\n", *a7, "",
          f"Maximum out-of-support exchange fraction in any A7 run: {oos_max:.3f} "
          "(R-SPRT-4 stop threshold 0.5).", ""]
    ready = not failed and not stop
    L += ["## Readiness\n",
          ("All Gates 1–9 pass and no hard-stop condition is triggered. Stage 4 still requires "
           "explicit user approval (spec §16)." if ready else
           "**NOT ready for Stage 4.** At least one gate failed or a hard-stop condition "
           "triggered. Per the stage rules: STOP, diagnose, and do not change the algorithm "
           "or tune parameters to make a gate pass."), ""]
    (ROOT / "MECHANISM_SMOKE_REPORT.md").write_text("\n".join(L))
    (ROOT / "results" / "n1_smoke").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "n1_smoke" / "gates.json").write_text(
        json.dumps({str(g): {"pass": ok, "evidence": ev} for g, (ok, ev) in gates.items()} | {"stop": stop}, indent=2))
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
