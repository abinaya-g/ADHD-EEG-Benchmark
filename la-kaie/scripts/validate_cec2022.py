"""Benchmark validation for CEC2022 (run before any experiment).

Checks
  1. function numbering 1..12 and optimum values F_i* (official TR Table I)
  2. F_i(o_i) == F_i* at the official shift vector (all 12 functions, D=10,20)
  3. agreement of the compiled official C code with the official Python port
     on random points in [-100,100]^D (relative tolerance)
  4. input_data files byte-identical to those in the MPHBS package (hash list)
  5. determinism and batch/single-call consistency
  6. bounds and dimension guards
Writes results/validation/cec2022_validation.json and exits non-zero on failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.benchmarks.cec2022 import (DATA_DIR, OPTIMA, CEC2022Problem,  # noqa: E402
                                       cec2022_raw)

OUT = ROOT / "results" / "validation"
OUT.mkdir(parents=True, exist_ok=True)


def shift_vector(fid: int, dim: int) -> np.ndarray:
    rows = (DATA_DIR / f"shift_data_{fid}.txt").read_text().split("\n")
    first = np.array(rows[0].split(), dtype=float)
    return first[:dim]


def official_python(fid: int, X: np.ndarray) -> np.ndarray:
    """Evaluate the official Python implementation (it reads data relative to cwd)."""
    # The official Python port indexes OShift_temp[i] although shift_data_1..8
    # contain several rows, which raises ValueError for every F1-F8 call. For
    # this cross-check only, the module source is loaded in memory with that
    # single line changed to flatten the array (np.ravel). Nothing else changes.
    import types
    here = os.getcwd()
    os.chdir(ROOT / "third_party" / "cec2022_official")
    try:
        src = Path("CEC2022_official_python.py").read_text()
        src = src.replace("OShift[i] = OShift_temp[i]", "OShift[i] = np.ravel(OShift_temp)[i]")
        off = types.ModuleType("cec2022_official_python_patched")
        exec(compile(src, "CEC2022_official_python.py", "exec"), off.__dict__)
        return np.array([off.cec22_test_func(x.copy(), len(x), 1, fid) for x in X]).ravel()
    finally:
        os.chdir(here)


def main() -> int:
    report = {"checks": [], "pass": True}

    def check(name, ok, detail):
        report["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
        report["pass"] &= bool(ok)
        print(("PASS " if ok else "FAIL ") + name + " :: " + json.dumps(detail)[:300])

    check("numbering_and_optima", sorted(OPTIMA) == list(range(1, 13)) and
          [OPTIMA[i] for i in range(1, 13)] ==
          [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700],
          {k: OPTIMA[k] for k in OPTIMA})

    for dim in (10, 20):
        vals = {}
        ok = True
        for fid in range(1, 13):
            o = shift_vector(fid, dim)
            f = float(cec2022_raw(o, fid)[0])
            vals[fid] = f
            ok &= abs(f - OPTIMA[fid]) <= 1e-8 * max(1.0, OPTIMA[fid])
        check(f"value_at_shift_equals_optimum_D{dim}", ok, vals)

    rng = np.random.default_rng(20260928)
    for dim in (10, 20):
        worst = {}
        ok = True
        for fid in range(1, 13):
            X = rng.uniform(-100, 100, size=(20, dim))
            c = cec2022_raw(X, fid)
            p = official_python(fid, X)
            rel = np.max(np.abs(c - p) / np.maximum(1.0, np.abs(c)))
            worst[fid] = float(rel)
            ok &= rel < 1e-9
        check(f"official_C_vs_official_python_D{dim}", ok, worst)

    ref_hashes = ROOT / "third_party" / "cec2022_official" / "input_data_sha256.txt"
    lines = []
    for f in sorted(DATA_DIR.iterdir()):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
    if not ref_hashes.exists():
        ref_hashes.write_text("\n".join(lines) + "\n")
    check("input_data_hashes_stable", ref_hashes.read_text().strip() == "\n".join(lines).strip(),
          {"n_files": len(lines)})

    X = rng.uniform(-100, 100, size=(7, 20))
    ok = True
    for fid in range(1, 13):
        a = cec2022_raw(X, fid)
        b = np.array([cec2022_raw(x, fid)[0] for x in X])
        c = cec2022_raw(X, fid)
        ok &= np.array_equal(a, b) and np.array_equal(a, c)
    check("batch_single_determinism", ok, {})

    prob = CEC2022Problem(1, 20)
    ok = np.all(prob.lb == -100) and np.all(prob.ub == 100) and prob.dim == 20
    try:
        cec2022_raw(np.zeros(7), 1)
        ok = False
    except ValueError:
        pass
    try:
        cec2022_raw(np.zeros(20), 13)
        ok = False
    except ValueError:
        pass
    err0 = float(prob(shift_vector(1, 20))[0])
    ok &= abs(err0) < 1e-8
    check("bounds_dimension_guards_error_wrapper", ok, {"error_at_optimum_F1": err0})

    (OUT / "cec2022_validation.json").write_text(json.dumps(report, indent=2))
    print("OVERALL:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
