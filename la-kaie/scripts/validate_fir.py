"""Validation of the FIR benchmark objective (MPHBS paper Eqs. (98)-(108)).

Checks
  1. the 8 YAML cases equal an independent transcription of the paper's list
  2. FFT-based objective == direct DTFT summation (independent implementation)
  3. analytic values: h=0 -> F=Wp ; h=delta -> F=Ws ; F(-h)=F(h); F(reverse h)=F(h)
  4. band index sets: sizes / edges follow Eqs. (102)-(105) on w_n=n/1024
  5. a classical windowed-sinc design (scipy.signal.firwin) scores far below random filters
  6. gray-box quadratic stopband form reproduces MSEs
Writes results/validation/fir_validation.json; non-zero exit on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.benchmarks.fir import FIRProblem, load_cases  # noqa: E402

# Independent transcription of the paper text (Sec. 5.1.2 bullet list + Ws rule)
PAPER = {
    1: ("LPF_Easy", "LPF", {"Fp": 0.18, "Fs": 0.25}, 100),
    2: ("LPF_Hard", "LPF", {"Fp": 0.22, "Fs": 0.26}, 120),
    3: ("HPF_Easy", "HPF", {"Fs": 0.75, "Fp": 0.82}, 100),
    4: ("HPF_Hard", "HPF", {"Fs": 0.74, "Fp": 0.78}, 120),
    5: ("BPF_Wide", "BPF", {"Fs1": 0.25, "Fp1": 0.35, "Fp2": 0.65, "Fs2": 0.75}, 100),
    6: ("BPF_Narrow", "BPF", {"Fs1": 0.32, "Fp1": 0.40, "Fp2": 0.60, "Fs2": 0.68}, 120),
    7: ("BSF_Wide", "BSF", {"Fp1": 0.25, "Fs1": 0.35, "Fs2": 0.65, "Fp2": 0.75}, 100),
    8: ("BSF_Narrow", "BSF", {"Fp1": 0.32, "Fs1": 0.40, "Fs2": 0.60, "Fp2": 0.68}, 120),
}


def direct_objective(prob: FIRProblem, h: np.ndarray) -> float:
    n = np.arange(prob.dim)
    omega = np.pi * prob.w
    H = np.exp(-1j * np.outer(omega, n)) @ h
    mse_s = np.mean(np.abs(H[prob.S]) ** 2)
    mse_p = np.mean((np.abs(H[prob.P]) - 1) ** 2)
    return prob.Wp * mse_p + prob.Ws * mse_s


def main() -> int:
    report = {"checks": [], "pass": True}

    def check(name, ok, detail):
        report["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
        report["pass"] &= bool(ok)
        print(("PASS " if ok else "FAIL ") + name + " :: " + json.dumps(detail)[:260])

    spec = load_cases()
    c = spec["common"]
    ok = (c["filter_order"] == 30 and c["dimension"] == 31 and c["n_fft"] == 2048 and
          c["lower_bound"] == -1 and c["upper_bound"] == 1 and c["Wp"] == 1 and len(spec["cases"]) == 8)
    for case in spec["cases"]:
        name, typ, edges, ws = PAPER[case["id"]]
        ok &= case["name"] == name and case["type"] == typ and case["Ws"] == ws
        ok &= all(abs(case[k] - v) < 1e-15 for k, v in edges.items())
    check("cases_match_paper_transcription", ok, {"n_cases": len(spec["cases"])})

    rng = np.random.default_rng(7)
    worst = 0.0
    for cid in range(1, 9):
        p = FIRProblem(cid, spec)
        H = rng.uniform(-1, 1, size=(10, 31))
        f_fft = p(H)
        f_dir = np.array([direct_objective(p, h) for h in H])
        worst = max(worst, float(np.max(np.abs(f_fft - f_dir) / np.abs(f_dir))))
    check("fft_equals_direct_dtft", worst < 1e-10, {"max_rel_err": worst})

    ok = True
    vals = {}
    for cid in range(1, 9):
        p = FIRProblem(cid, spec)
        z = float(p(np.zeros(31))[0])
        d = np.zeros(31); d[0] = 1.0
        fd = float(p(d)[0])
        h = rng.uniform(-1, 1, 31)
        ok &= abs(z - p.Wp) < 1e-12 and abs(fd - p.Ws) < 1e-9
        ok &= abs(p(h)[0] - p(-h)[0]) < 1e-12 and abs(p(h)[0] - p(h[::-1])[0]) < 1e-10
        vals[cid] = {"F(0)": z, "F(delta)": fd}
    check("analytic_values_and_invariances", ok, vals)

    sizes = {}
    ok = True
    for cid in range(1, 9):
        p = FIRProblem(cid, spec)
        sizes[cid] = {"|P|": int(p.P.size), "|S|": int(p.S.size)}
        ok &= p.P.size > 0 and p.S.size > 0 and len(np.intersect1d(p.P, p.S)) == 0
    p1 = FIRProblem(1, spec)   # LPF_Easy: w<=0.18 -> n<=184 ; w>=0.25 -> n>=256
    ok &= p1.P.size == 185 and p1.S.size == 1025 - 256 and p1.P[-1] == 184 and p1.S[0] == 256
    check("band_index_sets", ok, sizes)

    ok = True
    designs = {}
    for cid in range(1, 9):
        p = FIRProblem(cid, spec)
        cs = p.case
        t = cs["type"]
        if t == "LPF":
            h = signal.firwin(31, (cs["Fp"] + cs["Fs"]) / 2)
        elif t == "HPF":
            h = signal.firwin(31, (cs["Fp"] + cs["Fs"]) / 2, pass_zero=False)
        elif t == "BPF":
            h = signal.firwin(31, [(cs["Fs1"] + cs["Fp1"]) / 2, (cs["Fp2"] + cs["Fs2"]) / 2], pass_zero=False)
        else:
            h = signal.firwin(31, [(cs["Fp1"] + cs["Fs1"]) / 2, (cs["Fs2"] + cs["Fp2"]) / 2])
        fw = float(p(h)[0])
        fr = float(np.median(p(rng.uniform(-1, 1, size=(200, 31)))))
        designs[cid] = {"firwin": fw, "median_random": fr}
        ok &= fw < 0.05 * fr
    check("classical_design_sanity", ok, designs)

    ok = True
    for cid in range(1, 9):
        p = FIRProblem(cid, spec)
        Q = p.stopband_quadratic_form()
        H = rng.uniform(-1, 1, size=(5, 31))
        _, ms = p.components(H)
        ok &= np.allclose(np.einsum("ij,jk,ik->i", H, Q, H), ms, rtol=1e-9)
    check("stopband_quadratic_form", ok, {})

    out = ROOT / "results" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fir_validation.json").write_text(json.dumps(report, indent=2))
    print("OVERALL:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
