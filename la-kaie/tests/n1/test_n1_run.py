"""Run-level tests: T-FE, T-SHADOW, T-PREQ (log order), T-INIT, window integrity."""
import inspect

import numpy as np
import pytest

import n1.algorithm as alg
from lakaie.algorithms.hybrid_common import HybridState
from lakaie.core import CountedObjective, Recorder
from n1.config import ARMS, N1Config
from n1.synthetic import SyntheticProblem

CFG = N1Config()


def _run(arm, fid=1, D=10, seed=70000011, max_fe=20000):
    prob = SyntheticProblem(fid, D, run_seed=seed)
    obj = CountedObjective(prob, max_fe)
    rec = Recorder(prob.lb, prob.ub)
    out = alg.run_n1(prob, ARMS[arm], CFG, seed, obj, max_fe, rec)
    return out, obj, rec


@pytest.mark.parametrize("arm", ["A0", "A7", "A8", "A1", "A2", "A3", "A5", "A6", "A11", "P", "A10"])
def test_fe_accounting_exact(arm):
    """T-FE: per-phase FE sum == counter; budget never exceeded; bounded unused FE."""
    out, obj, _ = _run(arm)
    s = out.summary
    assert s["fe_audit_ok"] and sum(s["fe_by_tag"].values()) == obj.fe
    assert obj.fe <= 20000 and s["fe_unused_ok"]
    assert s["fe_by_tag"]["init"] == 2 * CFG.N
    n = s["n_iter"]
    assert s["fe_by_tag"]["phase1"] == 2 * CFG.N * n and s["fe_by_tag"]["fad"] == CFG.N * n
    n_x = s["fe_by_tag"].get("exchange", 0) + s["fe_by_tag"].get("probe", 0)
    assert n_x == len(out.arrays.get("x_y", []))
    if arm == "A0":
        assert n_x == 0
    assert s["n_hidden_checks"] == n


def test_hidden_evaluation_is_detected(monkeypatch):
    """T-FE: an objective call inside the zero-FE selection section is a hard error."""
    prob = SyntheticProblem(1, 10)
    obj = CountedObjective(prob, 20000)
    real = alg.evaluate_selector

    def leaky(*a, **k):
        obj(np.zeros((1, 10)))          # hidden evaluation
        return real(*a, **k)
    monkeypatch.setattr(alg, "evaluate_selector", leaky)
    with pytest.raises(alg.HiddenEvaluation):
        alg.run_n1(prob, ARMS["A0"], CFG, 1, obj, 20000, None)


def test_shadow_neutrality():
    """T-SHADOW: A0 with and without shadow selectors is bit-identical."""
    a, oa, ra = _run("A0", fid=3)
    b, ob, rb = _run("A0_noshadow", fid=3)
    assert oa.fe == ob.fe and oa.best == ob.best
    np.testing.assert_array_equal(oa.curve, ob.curve)
    for k in ("best", "current_mean", "current_median", "diversity"):
        np.testing.assert_array_equal(ra.as_arrays()[k], rb.as_arrays()[k])
    assert len(a.arrays["s_selector"]) > 0 and "s_selector" not in b.arrays


def test_p0_logged_before_evaluation():
    """T-PREQ (log order): every exchange record has fe_eval == fe_p0 + 1."""
    out, _, _ = _run("A7")
    x = out.arrays
    assert len(x["x_y"]) > 0
    assert np.all(x["x_fe_eval"] == x["x_fe_p0"] + 1)


def test_no_exchange_rows_in_windows():
    """Structural windows hold exactly the native successes (no exchange rows)."""
    out, _, _ = _run("A7")
    ops, nsucc = out.arrays["na_op"], out.arrays["na_n_succ"]
    nH = int(nsucc[ops == "HBA"].sum())
    nM = int(nsucc[(ops == "MPA") | (ops == "FAD")].sum())
    s = out.summary
    assert s["n_nonfinite_disp"] == 0
    assert s["n_window_rows_H"] == nH
    assert s["n_window_rows_M"] == nM
    assert int(out.arrays["x_y"].sum()) > 0          # exchanges were accepted, yet not recorded


def test_windows_reject_non_native():
    from n1.records import StructuralWindow
    w = StructuralWindow(3, 5)
    with pytest.raises(ValueError):
        w.append_native(np.ones((1, 3)), "EXCH", 0)


def test_initial_populations_identical_to_mphbs():
    """T-INIT: N1 rng_init reproduces MPHBS's HybridState initialisation."""
    prob = SyntheticProblem(2, 20)
    seed = 20260405 + 1001
    a = HybridState(CountedObjective(prob, None), prob.lb, prob.ub, 20, 30, np.random.default_rng(seed))
    rng_init = alg.spawn_streams(seed)[0]
    b = alg.make_state(CountedObjective(prob, None), prob.lb, prob.ub, 20, 30, rng_init)
    np.testing.assert_array_equal(a.X_H, b.X_H)
    np.testing.assert_array_equal(a.X_M, b.X_M)
    # the S7 box-initialiser replicates __init__ exactly when given the full box
    c = alg.make_state(CountedObjective(prob, None), prob.lb, prob.ub, 20, 30, np.random.default_rng(seed),
                       {"H": (prob.lb, prob.ub), "M": (prob.lb, prob.ub)})
    np.testing.assert_array_equal(a.X_H, c.X_H)
    np.testing.assert_array_equal(a.X_M, c.X_M)
    assert a.Pg == c.Pg


def test_s7_initial_boxes():
    prob = SyntheticProblem(7, 10)
    st = alg.make_state(CountedObjective(prob, None), prob.lb, prob.ub, 10, 30, np.random.default_rng(1),
                        prob.init_boxes)
    assert np.all(np.abs(st.X_H - prob.oA) <= 20 + 1e-12)
    assert np.all(np.abs(st.X_M - prob.oB) <= 20 + 1e-12)


def test_determinism():
    a, oa, _ = _run("A7", seed=5)
    b, ob, _ = _run("A7", seed=5)
    assert oa.best == ob.best
    np.testing.assert_array_equal(a.arrays["x_f_new"], b.arrays["x_f_new"])


def test_joint_half_is_offline_only():
    """JOINT-HALF is never computed inside the optimisation loop."""
    src = inspect.getsource(alg)
    assert "joint_half" not in src.lower()
