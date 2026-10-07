"""Amendment 2 (SHE-v2) tests: Stage 1 path bit-identical with nested=False, nesting of
the feature sets, and passivity of the B-2 zero-variance instrumentation."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from lakaie.core import CountedObjective, Recorder, make_checkpoints
from lakaie.she.algorithm import run_she
from lakaie.she.config import ROOT, VARIANTS, config_for
from lakaie.she.evidence import Evidence, features
from n1.synthetic import SyntheticProblem

STAGE1 = ROOT / "results" / "raw" / "she_diag"


def _rerun(variant, fid, D, seed, max_fe):
    pr = SyntheticProblem(fid, D, run_seed=seed)
    obj = CountedObjective(pr, max_fe, make_checkpoints(max_fe))
    out = run_she(pr, variant, config_for("SYN"), seed, obj, max_fe, Recorder(pr.lb, pr.ub))
    return out, obj


@pytest.mark.parametrize("vname,cell", [("SHE-NoE3", "S2_D10"), ("SHE-Uniform", "S1_D10")])
def test_stage1_bit_identical_against_stored_results(vname, cell):
    """nested=False (Stage 1 variants) reproduces the STORED Stage 1 run exactly:
    same pi, same losses, same choices/outcomes, same FE counts, same curve."""
    jp = STAGE1 / vname / cell / "run01.json"
    if not jp.exists():
        pytest.skip("Stage 1 raw data not present")
    rec = json.loads(jp.read_text())
    ref = np.load(jp.with_suffix(".npz"))
    assert VARIANTS[vname].nested is False
    out, obj = _rerun(VARIANTS[vname], rec["fid"], rec["dimension"], rec["seed"], rec["max_fe"])
    for k in ("r_pi", "r_loss", "r_gen", "r_y", "r_cost", "r_fe", "r_don", "r_pis", "A_H", "A_M"):
        assert np.array_equal(out.arrays[k], ref[k]), k
    assert np.array_equal(obj.curve, ref["curve"], equal_nan=True)
    assert obj.fe == rec["FE"]
    for k in ("exchange_fe", "h0_attempts", "records", "iters"):
        assert out.summary[k] == rec["summary"][k], k
    assert "zv_H" not in out.arrays                     # B-2 logging is off on the Stage 1 path


def test_nested_features_contain_h0():
    rng = np.random.default_rng(0)
    z1, z3 = rng.standard_normal((7, 5)), rng.standard_normal((7, 5))
    pairs = np.array([[0, 1], [2, 4]])
    f0 = features(0, z1, z3, pairs, nested=True)
    for g in (1, 2, 3):
        fn = features(g, z1, z3, pairs, nested=True)
        fo = features(g, z1, z3, pairs, nested=False)
        assert np.array_equal(fn[:, :2], f0)                 # h0 columns are a prefix
        assert np.array_equal(np.delete(fn, 1, axis=1), fo)  # nothing else changed
    assert np.array_equal(features(0, z1, z3, pairs, nested=False), f0)


def test_nested_evidence_dimensions():
    cfg = config_for("SYN")
    ev = Evidence(cfg, 6, nested=True)
    assert ev.models[1].theta.size == 2 * 6 + 2
    ev.set_pairs(np.array([[0, 1]]))
    assert ev.models[2].theta.size == 2 * 6 + 3
    assert ev.models[0].theta.size == 2


def test_zv_logging_is_passive():
    v = VARIANTS["SHE-v2-NoE3"]
    a, oa = _rerun(v, 1, 10, 75500999, 8000)
    b, ob = _rerun(replace(v, zv_log=False), 1, 10, 75500999, 8000)
    for k in ("r_pi", "r_loss", "r_gen", "r_y", "A_H", "A_M"):
        assert np.array_equal(a.arrays[k], b.arrays[k]), k
    assert np.array_equal(oa.curve, ob.curve, equal_nan=True)
    assert "zv_H" in a.arrays and a.arrays["zv_H"].size == a.arrays["A_H_n"].size
    assert a.summary["fe_audit_ok"] and a.summary["exact_E_ok"]
