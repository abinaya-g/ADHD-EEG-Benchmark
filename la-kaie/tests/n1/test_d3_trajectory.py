"""T-D3-TRAJECTORY: the D3 diagnostic logging is purely observational.

The same run with diag=None (production logging) and diag=D3Recorder must be
bit-identical. Tolerance: none (exact equality everywhere).

The objective is wrapped to record the full evaluation sequence (every row X and
its value f, in call order). Populations are formed only from evaluated points
under deterministic acceptance rules, so an identical evaluation sequence
(same X, same f, same order, same length) implies identical HBA and MPA
populations at every iteration. The recorded populations are additionally
checked to consist of evaluated points.
"""
import numpy as np
import pytest

import n1.algorithm as alg
from lakaie.core import CountedObjective, Recorder
from n1.config import ARMS, N1Config
from n1.diagnostics import D3Recorder
from n1.synthetic import SyntheticProblem

CFG = N1Config()


class Tape:
    def __init__(self, fn):
        self.fn, self.X, self.f = fn, [], []

    def __call__(self, X):
        f = self.fn(X)
        self.X.append(np.array(X, copy=True)); self.f.append(np.array(f, copy=True))
        return f


def _run(arm, fid, D, seed, max_fe, diag):
    prob = SyntheticProblem(fid, D, run_seed=seed)
    tape = Tape(prob)
    obj = CountedObjective(tape, max_fe)
    rec = Recorder(prob.lb, prob.ub)
    out = alg.run_n1(prob, ARMS[arm], CFG, seed, obj, max_fe, rec, diag=diag)
    return out, obj, rec, tape


@pytest.mark.parametrize("arm,fid,D", [("A0", 1, 10), ("A0", 2, 20), ("A0", 3, 10), ("A7", 3, 10)])
def test_d3_logging_is_trajectory_neutral(arm, fid, D):
    seed, max_fe = 81200000 + 1000 * (10 * fid + 1) + 1, 12000
    a, oa, ra, ta = _run(arm, fid, D, seed, max_fe, None)
    d = D3Recorder(CFG.N, D)
    b, ob, rb, tb = _run(arm, fid, D, seed, max_fe, d)
    # 1 FE sequence and 6 termination point
    assert oa.fe == ob.fe and len(ta.X) == len(tb.X)
    assert [x.shape[0] for x in ta.X] == [x.shape[0] for x in tb.X]
    assert a.summary["n_iter"] == b.summary["n_iter"]
    # 2 objective values and evaluated points (bitwise)
    for xa, xb, fa, fb in zip(ta.X, tb.X, ta.f, tb.f):
        assert np.array_equal(xa, xb) and np.array_equal(fa, fb)
    # 5 final objective, best-fitness trajectory
    assert oa.best == ob.best
    for k, v in ra.as_arrays().items():
        np.testing.assert_array_equal(v, rb.as_arrays()[k])
    # production logs unchanged
    assert set(a.arrays) == set(b.arrays)
    for k in a.arrays:
        np.testing.assert_array_equal(a.arrays[k], b.arrays[k])
    # 3/4 populations: recorded HBA/MPA populations consist of evaluated points
    ev = np.vstack(tb.X)
    arr = d.arrays()
    for key in ("d3_ep_XH", "d3_ep_XM"):
        P = arr[key].reshape(-1, D)
        assert P.shape[0] > 0
        hits = (ev[None, :, :] == P[:, None, :]).all(axis=2).any(axis=1)
        assert hits.all()


def test_d3_fields_are_auditable():
    d = D3Recorder(CFG.N, 10)
    out, obj, rec, tape = _run("A0", 2, 10, 81200000 + 1000 * 21 + 2, 12000, d)
    a = d.arrays()
    n_it = out.summary["n_iter"]
    assert a["d3_it_t"].size == n_it == a["d3_it_scale"].shape[0] == a["d3_it_CF"].size
    # standardized = raw / scale (the production transformation), exactly
    s = a["d3_it_scale"][a["d3_row_t"]]
    np.testing.assert_array_equal(a["d3_row_std"], a["d3_row_raw"] / s)
    # the standardized rows are exactly the rows production appended to the windows
    Zprod = np.concatenate([out.arrays["streamH_rows"], out.arrays["streamM_rows"]]).astype(np.float32)
    Zd3 = np.concatenate([a["d3_row_std"][a["d3_row_pop"] == 0], a["d3_row_std"][a["d3_row_pop"] == 1]])
    np.testing.assert_array_equal(np.sort(Zprod, axis=0), np.sort(Zd3.astype(np.float32), axis=0))
    # agent IDs are valid row indices; one success row per (t, op, agent) at most
    assert a["d3_row_agent"].min() >= 0 and a["d3_row_agent"].max() < CFG.N
    key = a["d3_row_t"].astype(np.int64) * 100000 + a["d3_row_op"].astype(np.int64) * 1000 + a["d3_row_agent"].astype(np.int64)
    assert np.unique(key).size == key.size
    # every attempt logged: 3N attempts per iteration
    assert a["d3_att_y"].size == 3 * CFG.N * n_it
    # epoch covariance consistent with the logged population
    X = a["d3_ep_XH"][0]
    np.testing.assert_allclose(a["d3_ep_H_cov"][0], np.cov(X.T, ddof=0), rtol=1e-12, atol=1e-12)
